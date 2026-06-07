import { NextResponse } from "next/server";
import { Chess } from "chess.js";
import { getModelSession } from "@/lib/onnx";
import { createResNetTensor } from "@/lib/tensor"; // <--- ОБНОВЛЕНО

const evalCache = new Map<string, number>();

async function evaluatePosition(fen: string, session: any): Promise<number> {
	// Убираем счетчики полуходов для кэширования
	const cleanFen = fen.split(" ").slice(0, 4).join(" ");
	if (evalCache.has(cleanFen)) return evalCache.get(cleanFen)!;

	// Создаем правильный 3D тензор
	const feeds = createResNetTensor(fen);
	const results = await session.run(feeds);

	// Читаем две головы из Трансформера
	const rawScore = (results["score"].data as Float32Array)[0];
	const mateLogit = (results["mate"].data as Float32Array)[0];

	const mateProb = 1 / (1 + Math.exp(-mateLogit));

	let finalScore = rawScore;

	// Матовый приоритет
	if (mateProb > 0.85) {
		finalScore = rawScore > 0 ? 30.0 : -30.0;
	}

	evalCache.set(cleanFen, finalScore);
	return finalScore;
}

// Упорядочивание ходов (КРИТИЧЕСКИ ВАЖНО ДЛЯ АЛЬФА-БЕТЫ)
// Если мы сначала смотрим хорошие ходы, альфа-бета отсекает 90% дерева
function orderMoves(game: Chess, moves: string[]) {
	return moves.sort((a, b) => {
		let scoreA = 0,
			scoreB = 0;
		if (a.includes("x")) scoreA += 10; // Взятия проверяем первыми
		if (a.includes("+")) scoreA += 5; // Шахи
		if (b.includes("x")) scoreB += 10;
		if (b.includes("+")) scoreB += 5;
		return scoreB - scoreA;
	});
}

// Классический Alpha-Beta поиск
async function alphaBeta(
	game: Chess,
	depth: number,
	alpha: number,
	beta: number,
	isWhite: boolean,
	session: any,
): Promise<number> {
	if (depth === 0 || game.isGameOver()) {
		if (game.isCheckmate())
			// ВАЖНО: + depth заставляет движок предпочитать БЫСТРЫЕ маты
			// MateScore (10000) перебивает любые пешки (30.0)
			return isWhite ? -(10000 + depth) : 10000 + depth;
		if (game.isDraw()) return 0;

		if (depth === 0) {
			return await evaluatePosition(game.fen(), session);
		}

		// В идеале здесь должен быть Quiescence Search,
		// но для начала просто вызываем нейросеть
		return await evaluatePosition(game.fen(), session);
	}

	const moves = game.moves();
	orderMoves(game, moves); // Сортируем ходы для максимального отсечения

	if (isWhite) {
		let maxEval = -Infinity;
		for (const move of moves) {
			game.move(move);
			const ev = await alphaBeta(
				game,
				depth - 1,
				alpha,
				beta,
				false,
				session,
			);
			game.undo();

			maxEval = Math.max(maxEval, ev);
			alpha = Math.max(alpha, ev);
			if (beta <= alpha) break; // Отсечение ветки!
		}
		return maxEval;
	} else {
		let minEval = Infinity;
		for (const move of moves) {
			game.move(move);
			const ev = await alphaBeta(
				game,
				depth - 1,
				alpha,
				beta,
				true,
				session,
			);
			game.undo();

			minEval = Math.min(minEval, ev);
			beta = Math.min(beta, ev);
			if (beta <= alpha) break; // Отсечение ветки!
		}
		return minEval;
	}
}

async function getBestMove(fen: string, depth: number, session: any) {
	const game = new Chess(fen);
	const moves = game.moves();
	let bestMove = moves[0];
	const isWhite = game.turn() === "w";

	let bestValue = isWhite ? -Infinity : Infinity;

	for (const move of moves) {
		game.move(move);
		// Запускаем поиск для ветки
		const boardValue = await alphaBeta(
			game,
			depth - 1,
			-Infinity,
			Infinity,
			!isWhite,
			session,
		);
		game.undo();

		if (isWhite) {
			if (boardValue > bestValue) {
				bestValue = boardValue;
				bestMove = move;
			}
		} else {
			if (boardValue < bestValue) {
				bestValue = boardValue;
				bestMove = move;
			}
		}
	}

	return { move: bestMove, score: bestValue };
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
			console.log(`\n🧠 [Smart Search] Start | Depth: ${searchDepth}`);
			const startTime = Date.now();

			bestMoveResult = await getBestMove(fen, searchDepth, session);

			console.log(
				`⏱️ [Smart Search] Zakończono w ${Date.now() - startTime}ms`,
			);
		} else {
			bestMoveResult = await getBestMove(fen, 1, session);
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
