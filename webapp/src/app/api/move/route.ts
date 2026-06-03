import { NextResponse } from "next/server";
import { Chess } from "chess.js";
import { getModelSession } from "@/lib/onnx";
import { createBatchTensor, decodeEvaluations } from "@/lib/tensor";
import { Tensor } from "onnxruntime-node";

export async function POST(request: Request) {
	try {
		const { fen } = await request.json();

		if (!fen || typeof fen !== "string") {
			return NextResponse.json(
				{ error: "Invalid FEN provided" },
				{ status: 400 },
			);
		}

		const chess = new Chess(fen);
		const legalMoves = chess.moves({ verbose: true });

		if (legalMoves.length === 0) {
			return NextResponse.json({
				bestMove: null,
				message: "No legal moves available",
			});
		}

		// 1. Инициализируем сессию
		const session = await getModelSession();

		// 2. Генерируем FEN для каждого возможного хода
		const futureFens = legalMoves.map((move) => {
			const tempChess = new Chess(fen);
			tempChess.move(move);
			return tempChess.fen();
		});

		console.log(
			`\n[Inference] Evaluating ${futureFens.length} moves concurrently...`,
		);

		// 3. Запускаем параллельные вычисления
        const promises = futureFens.map(async (f) => {
            const inputTensor = createBatchTensor([f]); 
            const feeds: Record<string, Tensor> = { [session.inputNames[0]]: inputTensor };
            
            const results = await session.run(feeds);
            const logits = results[session.outputNames[0]].data as Float32Array;
            
            let score = decodeEvaluations(logits, 1)[0];

            // 🎯 ВОТ ИСПРАВЛЕНИЕ: ПРИВОДИМ ОЦЕНКУ К АБСОЛЮТУ (От лица Белых)
            // Если ход передает очередь черным (' b '), нейросеть выдала оценку в пользу черных.
            // Переворачиваем знак, чтобы всегда было: + Белые выигрывают, - Черные выигрывают.
            if (f.includes(' b ')) {
                score = -score;
            }

            return score;
        });

		// Ждем, пока все 20+ потоков завершат оценку
		const scores = await Promise.all(promises);

		// 4. Ищем лучший ход (Белые ищут максимум, Черные - минимум)
		const isWhite = chess.turn() === "w";
		let bestMoveIdx = 0;
		let bestScore = isWhite ? -Infinity : Infinity;

		for (let i = 0; i < scores.length; i++) {
			const score = scores[i];

			// Выводим оценки в консоль для интереса
			console.log(
				`Move: ${legalMoves[i].san.padEnd(5)} | Score: ${score.toFixed(2)} pawns`,
			);

			if (isWhite && score > bestScore) {
				bestScore = score;
				bestMoveIdx = i;
			} else if (!isWhite && score < bestScore) {
				bestScore = score;
				bestMoveIdx = i;
			}
		}

		const bestMove = legalMoves[bestMoveIdx];
		console.log(
			`>>> Best move selected: ${bestMove.san} (${bestScore.toFixed(2)})`,
		);

		return NextResponse.json({
			bestMove: bestMove.san,
			from: bestMove.from,
			to: bestMove.to,
			score: bestScore,
		});
	} catch (error) {
		console.error("Inference Error:", error);
		return NextResponse.json(
			{ error: "Internal Server Error" },
			{ status: 500 },
		);
	}
}
