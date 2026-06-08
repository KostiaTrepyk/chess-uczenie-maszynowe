import { NextResponse } from "next/server";
import { Chess, Move } from "chess.js";
import { getModelSession } from "@/lib/onnx";
import { createResNetTensor } from "@/lib/tensor";

const evalCache = new Map<string, number>();
const MAX_CACHE_SIZE = 150000;

// Утилита для получения "чистого" FEN (без счетчиков ходов),
// чтобы правильно определять повторения позиций.
function getCleanFen(fen: string) {
	return fen.split(" ").slice(0, 4).join(" ");
}

// Восстанавливаем все FEN-ы, которые были в партии
function getPastFens(history: string[]): string[] {
	const g = new Chess();
	const fens = [getCleanFen(g.fen())];
	for (const move of history) {
		try {
			g.move(move);
		} catch (e) {
			break;
		}
		fens.push(getCleanFen(g.fen()));
	}
	return fens;
}

async function evaluatePosition(fen: string, session: any): Promise<number> {
	if (evalCache.size > MAX_CACHE_SIZE) evalCache.clear();

	const cleanFen = getCleanFen(fen);
	if (evalCache.has(cleanFen)) return evalCache.get(cleanFen)!;

	const isWhiteTurn = fen.split(" ")[1] === "w";
	const feeds = createResNetTensor(fen);
	const results = await session.run(feeds);

	const rawScore = (results["score"].data as Float32Array)[0];
	const mateLogit = (results["mate"].data as Float32Array)[0];
	const mateProb = 1 / (1 + Math.exp(-mateLogit));

	let finalScore = rawScore;
	if (mateProb > 0.75) {
		const mateScore = 20.0 + mateProb * 10.0;
		finalScore = rawScore > 0 ? mateScore : -mateScore;
	} else {
		finalScore = Math.max(-15.0, Math.min(15.0, rawScore));
	}

	const absoluteScore = isWhiteTurn ? finalScore : -finalScore;
	evalCache.set(cleanFen, absoluteScore);
	return absoluteScore;
}

function getPieceValue(p?: string) {
	if (!p) return 0;
	if (p === "q") return 900;
	if (p === "r") return 500;
	if (p === "b" || p === "n") return 300;
	if (p === "p") return 100;
	return 0;
}

function orderMoves(moves: Move[]) {
	return moves.sort((a, b) => {
		let scoreA = 0,
			scoreB = 0;
		if (a.san.includes("#")) scoreA += 10000;
		if (b.san.includes("#")) scoreB += 10000;
		if (a.san.includes("+")) scoreA += 500;
		if (b.san.includes("+")) scoreB += 500;

		if (a.captured)
			scoreA += 1000 + getPieceValue(a.captured) - getPieceValue(a.piece);
		if (b.captured)
			scoreB += 1000 + getPieceValue(b.captured) - getPieceValue(b.piece);

		if (!a.captured) {
			if (a.piece === "q") scoreA += 50;
			else if (a.piece === "r") scoreA += 30;
			else if (a.piece === "n" || a.piece === "b") scoreA += 10;
		}
		if (!b.captured) {
			if (b.piece === "q") scoreB += 50;
			else if (b.piece === "r") scoreB += 30;
			else if (b.piece === "n" || b.piece === "b") scoreB += 10;
		}

		return scoreB - scoreA;
	});
}

async function quiescence(
	game: Chess,
	alpha: number,
	beta: number,
	isWhite: boolean,
	session: any,
	qsLimit: number = 3,
): Promise<number> {
	const standPat = await evaluatePosition(game.fen(), session);
	if (isWhite) {
		if (standPat >= beta) return beta;
		if (alpha < standPat) alpha = standPat;
	} else {
		if (standPat <= alpha) return alpha;
		if (beta > standPat) beta = standPat;
	}
	if (qsLimit === 0) return standPat;

	const moves = game
		.moves({ verbose: true })
		.filter((m) => m.captured || m.san.includes("="));
	if (moves.length === 0) return standPat;

	orderMoves(moves);

	if (isWhite) {
		let maxEval = standPat;
		for (const move of moves) {
			game.move(move.san);
			const ev = await quiescence(
				game,
				alpha,
				beta,
				false,
				session,
				qsLimit - 1,
			);
			game.undo();
			maxEval = Math.max(maxEval, ev);
			alpha = Math.max(alpha, ev);
			if (beta <= alpha) break;
		}
		return maxEval;
	} else {
		let minEval = standPat;
		for (const move of moves) {
			game.move(move.san);
			const ev = await quiescence(
				game,
				alpha,
				beta,
				true,
				session,
				qsLimit - 1,
			);
			game.undo();
			minEval = Math.min(minEval, ev);
			beta = Math.min(beta, ev);
			if (beta <= alpha) break;
		}
		return minEval;
	}
}

async function alphaBeta(
	game: Chess,
	depth: number,
	alpha: number,
	beta: number,
	isWhite: boolean,
	session: any,
): Promise<number> {
	if (game.isGameOver()) {
		if (game.isCheckmate())
			return isWhite ? -(100000 + depth) : 100000 + depth;
		if (game.isDraw()) return 0;
	}
	if (depth === 0)
		return await quiescence(game, alpha, beta, isWhite, session);

	const moves = game.moves({ verbose: true });
	orderMoves(moves);

	let limitedMoves: Move[] = [];
	let quietCount = 0;
	for (const m of moves) {
		if (
			m.san.includes("+") ||
			m.san.includes("#") ||
			m.captured ||
			m.san.includes("=")
		) {
			limitedMoves.push(m);
		} else if (quietCount < 15) {
			limitedMoves.push(m);
			quietCount++;
		}
	}

	if (isWhite) {
		let maxEval = -Infinity;
		for (const move of limitedMoves) {
			game.move(move.san);
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
			if (beta <= alpha) break;
		}
		return maxEval;
	} else {
		let minEval = Infinity;
		for (const move of limitedMoves) {
			game.move(move.san);
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
			if (beta <= alpha) break;
		}
		return minEval;
	}
}

// --- ИЗМЕНЕННАЯ ФУНКЦИЯ ---
async function getBestMove(
	fen: string,
	depth: number,
	session: any,
	pastFens: string[],
) {
	const game = new Chess(fen);
	const moves = game.moves({ verbose: true });
	orderMoves(moves);

	let limitedMoves: Move[] = [];
	let quietCount = 0;
	for (const m of moves) {
		if (
			m.san.includes("+") ||
			m.san.includes("#") ||
			m.captured ||
			m.san.includes("=")
		) {
			limitedMoves.push(m);
		} else if (quietCount < 20) {
			limitedMoves.push(m);
			quietCount++;
		}
	}

	if (limitedMoves.length === 0) return { move: null, score: 0 };
	const isWhite = game.turn() === "w";

	// 1. Собираем оценки ВСЕХ ходов
	let evaluatedMoves: { move: string; score: number }[] = [];
	for (const move of limitedMoves) {
		game.move(move.san);
		const boardValue = await alphaBeta(
			game,
			depth - 1,
			-Infinity,
			Infinity,
			!isWhite,
			session,
		);
		game.undo();
		evaluatedMoves.push({ move: move.san, score: boardValue });
	}

	// 2. Сортируем: для белых ищем максимум, для черных — минимум
	if (isWhite) {
		evaluatedMoves.sort((a, b) => b.score - a.score);
	} else {
		evaluatedMoves.sort((a, b) => a.score - b.score);
	}

	let selectedMove = evaluatedMoves[0];

	// 3. ЛОГИКА АНТИ-ПОВТОРЕНИЯ
	for (let i = 0; i < evaluatedMoves.length; i++) {
		const candidate = evaluatedMoves[i];

		// Защита от глупости:
		// Мы не хотим избегать ничьей ценой потери Ферзя.
		// Если альтернативный ход хуже лучшего больше чем на 1.5 пешки — сдаемся и повторяем позицию.
		if (Math.abs(evaluatedMoves[0].score - candidate.score) > 1.5) {
			break;
		}

		game.move(candidate.move);
		const newFen = getCleanFen(game.fen());
		game.undo();

		// Проверяем, был ли такой FEN в партии
		if (pastFens.includes(newFen)) {
			// Было! Этот ход ведет к танцам. Пропускаем его и берем следующий по списку!
			continue;
		} else {
			// Нашли хороший уникальный ход
			selectedMove = candidate;
			break;
		}
	}

	return { move: selectedMove.move, score: selectedMove.score };
}

export async function POST(req: Request) {
	try {
		const body = await req.json();
		const fen = body.fen;
		const history = body.history || []; // Читаем историю с фронта
		const searchMode = body.searchMode || "simple";
		const searchDepth = parseInt(body.searchDepth) || 1;

		const session = await getModelSession();
		const pastFens = getPastFens(history); // Конвертируем историю в FEN-ы
		let bestMoveResult;

		if (searchMode === "advanced" && searchDepth > 1) {
			console.log(`\n🧠 [Smart Search] Start | Depth: ${searchDepth}`);
			const startTime = Date.now();
			// Передаем pastFens внутрь поиска
			bestMoveResult = await getBestMove(
				fen,
				searchDepth,
				session,
				pastFens,
			);
			console.log(
				`⏱️ [Smart Search] Zakończono w ${Date.now() - startTime}ms`,
			);
		} else {
			bestMoveResult = await getBestMove(fen, 1, session, pastFens);
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
