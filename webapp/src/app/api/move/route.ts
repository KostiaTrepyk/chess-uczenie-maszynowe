import { NextResponse } from "next/server";
import { Chess } from "chess.js";
import { getModelSession } from "@/lib/onnx";
import { createBatchTensor, decodeEvaluations } from "@/lib/tensor";

// Глобальный кэш (очищается при перезапуске сервера, можно вынести внутрь POST для очистки на каждый ход)
const evalCache = new Map<string, number>();

async function evaluateBatch(fens: string[], session: any): Promise<number[]> {
	const scores = new Array(fens.length).fill(0);
	const fensToEvaluate: string[] = [];
	const indicesToEvaluate: number[] = [];

	// 1. Проверяем кэш
	for (let i = 0; i < fens.length; i++) {
		// Убираем счетчики полуходов из FEN, они не влияют на оценку доски
		const cleanFen = fens[i].split(" ").slice(0, 4).join(" ");
		if (evalCache.has(cleanFen)) {
			scores[i] = evalCache.get(cleanFen)!;
		} else {
			fensToEvaluate.push(fens[i]);
			indicesToEvaluate.push(i);
		}
	}

	// 2. Если есть неизвестные позиции - прогоняем через ИИ
	if (fensToEvaluate.length > 0) {
		const inputTensor = createBatchTensor(fensToEvaluate);
		const feeds = { [session.inputNames[0]]: inputTensor };

		const results = await session.run(feeds);
		const logits = results[session.outputNames[0]].data as Float32Array;

		const newScores = decodeEvaluations(logits, fensToEvaluate.length);

		// 3. Сохраняем результаты и пишем в кэш
		for (let j = 0; j < newScores.length; j++) {
			const originalIndex = indicesToEvaluate[j];
			const fen = fensToEvaluate[j];
			const cleanFen = fen.split(" ").slice(0, 4).join(" ");

			// Каноническое выравнивание
			const finalScore = fen.includes(" b ")
				? -newScores[j]
				: newScores[j];

			scores[originalIndex] = finalScore;
			evalCache.set(cleanFen, finalScore);
		}
	}

	return scores;
}

// 2. Рекурсивный алгоритм Beam Search (Лучевой минимакс)
async function beamSearch(
	fen: string,
	depth: number,
	beamWidth: number,
	session: any,
): Promise<{ move: any; score: number }> {
	const game = new Chess(fen);
	const isWhiteTurn = game.turn() === "w";
	const legalMoves = game.moves({ verbose: true });

	// Базовые случаи: конец игры или отсутствие ходов
	if (legalMoves.length === 0) {
		if (game.isCheckmate())
			return { move: null, score: isWhiteTurn ? -100 : 100 };
		return { move: null, score: 0 };
	}

	// Шаг 1: Генерируем все будущие позиции (ширина текущего узла)
	const nextFens = legalMoves.map((m) => {
		const temp = new Chess(game.fen());
		temp.move(m);
		return temp.fen();
	});

	// Шаг 2: Параллельно оцениваем их все в нейросети (1 запрос на слой)
	const scores = await evaluateBatch(nextFens, session);

	// Шаг 3: Сортируем ходы (Белые ищут максимум, Черные — минимум)
	const scoredMoves = legalMoves.map((move, i) => ({
		move,
		fen: nextFens[i],
		score: scores[i],
	}));

	scoredMoves.sort((a, b) =>
		isWhiteTurn ? b.score - a.score : a.score - b.score,
	);

	// Базовый случай 2: Достигли нужной глубины, возвращаем лучший локальный ход
	if (depth <= 1) {
		return { move: scoredMoves[0].move, score: scoredMoves[0].score };
	}

	// Шаг 4: BEAM SEARCH - отрезаем слабые ходы, оставляем только ТОП-K (beamWidth)
	const topMoves = scoredMoves.slice(0, beamWidth);

	let bestMove = topMoves[0].move;
	let bestScore = isWhiteTurn ? -Infinity : Infinity;

	// Шаг 5: Погружаемся вглубь только для самых перспективных веток
	for (const tm of topMoves) {
		const childResult = await beamSearch(
			tm.fen,
			depth - 1,
			beamWidth,
			session,
		);

		if (isWhiteTurn) {
			if (childResult.score > bestScore) {
				bestScore = childResult.score;
				bestMove = tm.move;
			}
		} else {
			if (childResult.score < bestScore) {
				bestScore = childResult.score;
				bestMove = tm.move;
			}
		}
	}

	return { move: bestMove, score: bestScore };
}

export async function POST(req: Request) {
	try {
		const body = await req.json();
		const fen = body.fen;
		const searchMode = body.searchMode || "simple";
		const searchDepth = parseInt(body.searchDepth) || 1;

		const session = await getModelSession();

		let bestMoveResult;

		if (searchMode === "advanced" && searchDepth > 1) {
			const BEAM_WIDTH = 3; // Константа: сколько лучших веток исследовать дальше
			console.log(
				`\n🧠 [Beam Search] Start | Depth: ${searchDepth} | BeamWidth: ${BEAM_WIDTH}`,
			);
			const startTime = Date.now();

			bestMoveResult = await beamSearch(
				fen,
				searchDepth,
				BEAM_WIDTH,
				session,
			);

			console.log(
				`⏱️ [Beam Search] Zakończono w ${Date.now() - startTime}ms`,
			);
		} else {
			// Быстрый режим (глубина 1, смотрит все 30 ходов)
			bestMoveResult = await beamSearch(fen, 1, 30, session);
		}

		return NextResponse.json({
			bestMove: bestMoveResult.move,
			score: bestMoveResult.score,
		});
	} catch (error: any) {
		console.error("API Error:", error);
		return NextResponse.json({ error: error.message }, { status: 500 });
	}
}
