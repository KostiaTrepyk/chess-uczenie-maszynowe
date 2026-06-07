import { NextResponse } from "next/server";
import { Chess } from "chess.js";
import { getModelSession } from "@/lib/onnx";
import { createResNetTensor } from "@/lib/tensor"; // <--- ОБНОВЛЕНО

const evalCache = new Map<string, number>();

async function evaluatePosition(fen: string, session: any): Promise<number> {
	const cleanFen = fen.split(" ").slice(0, 4).join(" ");
	if (evalCache.has(cleanFen)) return evalCache.get(cleanFen)!;

	const isWhiteTurn = fen.split(" ")[1] === "w";
	const feeds = createResNetTensor(fen);
	const results = await session.run(feeds);

	const rawScore = (results["score"].data as Float32Array)[0];
	const mateLogit = (results["mate"].data as Float32Array)[0];
	const mateProb = 1 / (1 + Math.exp(-mateLogit));

	let finalScore = rawScore;

	if (mateProb > 0.75) {
		// --- ЭФФЕКТ КОМПАСА ---
		// Если пахнет матом, мы ПОЛНОСТЬЮ игнорируем сырые пешки (чтобы избежать скачков 12..50).
		// Мы строим оценку только на уверенности сети в мате.
		// Вероятность (0.75...0.99) даст плавный рост оценки (27.5 ... 29.9).
		// Бот будет карабкаться по этой "горке" прямо к шее короля!
		const mateScore = 20.0 + mateProb * 10.0;
		finalScore = rawScore > 0 ? mateScore : -mateScore;
	} else {
		// В обычной игре (миттельшпиль) жестко срезаем галлюцинации до 15 пешек
		finalScore = Math.max(-15.0, Math.min(15.0, rawScore));
	}

	const absoluteScore = isWhiteTurn ? finalScore : -finalScore;
	evalCache.set(cleanFen, absoluteScore);
	return absoluteScore;
}

// Упорядочивание ходов (КРИТИЧЕСКИ ВАЖНО)
function orderMoves(moves: string[]) {
	return moves.sort((a, b) => {
		let scoreA = 0,
			scoreB = 0;
		if (a.includes("#")) scoreA += 1000;
		if (b.includes("#")) scoreB += 1000;
		if (a.includes("+")) scoreA += 50;
		if (b.includes("+")) scoreB += 50;
		if (a.includes("x")) scoreA += 10;
		if (b.includes("x")) scoreB += 10;
		return scoreB - scoreA;
	});
}

// Классический Alpha-Beta поиск + Beam Search
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
			return isWhite ? -(100000 + depth) : 100000 + depth;
		if (game.isDraw()) return 0;

		return await evaluatePosition(game.fen(), session);
	}

	const moves = game.moves();
	orderMoves(moves);

	// --- BEAM SEARCH (ЛУЧЕВОЙ ПОИСК) ---
	// Сокращаем ширину дерева: берем все агрессивные ходы + максимум 5 тихих
	let limitedMoves = [];
	let quietCount = 0;
	for (const m of moves) {
		if (m.includes("+") || m.includes("#") || m.includes("x")) {
			limitedMoves.push(m);
		} else if (quietCount < 5) {
			limitedMoves.push(m);
			quietCount++;
		}
	}

	if (isWhite) {
		let maxEval = -Infinity;
		for (const move of limitedMoves) {
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
			if (beta <= alpha) break; // Отсечение!
		}
		return maxEval;
	} else {
		let minEval = Infinity;
		for (const move of limitedMoves) {
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
			if (beta <= alpha) break; // Отсечение!
		}
		return minEval;
	}
}

async function getBestMove(fen: string, depth: number, session: any) {
	const game = new Chess(fen);
	const moves = game.moves();
	orderMoves(moves);

	// Применяем то же отсечение на верхнем уровне
	let limitedMoves = [];
	let quietCount = 0;
	for (const m of moves) {
		if (m.includes("+") || m.includes("#") || m.includes("x")) {
			limitedMoves.push(m);
		} else if (quietCount < 5) {
			limitedMoves.push(m);
			quietCount++;
		}
	}

	let bestMove = limitedMoves[0];
	const isWhite = game.turn() === "w";
	let bestValue = isWhite ? -Infinity : Infinity;

	for (const move of limitedMoves) {
		game.move(move);
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
