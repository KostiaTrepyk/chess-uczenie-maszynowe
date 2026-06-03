"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Chess, Square } from "chess.js";
import { Chessboard as OriginalChessboard } from "react-chessboard";

const Chessboard = OriginalChessboard;

type GameMode = "ai-black" | "ai-white" | "pvp";

export default function Home() {
	const [game, setGame] = useState(new Chess());
	const [gameMode, setGameMode] = useState<GameMode>("ai-black");
	const [isAiThinking, setIsAiThinking] = useState(false);
	const [aiScore, setAiScore] = useState<number | null>(null);

	const fetchLock = useRef(false);
	const isFetching = useRef(false);

	const currentFen = game.fen();
	const isWhiteTurn = game.turn() === "w";
	const isGameOver = game.isGameOver();

	const makeMove = useCallback(
		(
			move: { from: string; to: string; promotion?: string },
			currentGame: Chess,
		) => {
			try {
				const gameCopy = new Chess();
				gameCopy.loadPgn(currentGame.pgn());
				const result = gameCopy.move(move);
				if (result) {
					setGame(gameCopy);
					return gameCopy;
				}
			} catch (e: any) {
				return null;
			}
			return null;
		},
		[],
	);

	const fetchAiMove = useCallback(
		async (currentGame: Chess) => {
			if (isFetching.current) return;
			isFetching.current = true;
			setIsAiThinking(true);

			try {
				const response = await fetch("/api/move", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ fen: currentGame.fen() }),
				});
				const data = await response.json();

				if (data.bestMove) {
					makeMove(data.bestMove, currentGame);
					setAiScore(data.score);
				}
			} catch (error) {
				console.error("🚨 API Error:", error);
			} finally {
				setIsAiThinking(false);
				isFetching.current = false;
			}
		},
		[makeMove],
	);

	useEffect(() => {
		if (isGameOver) return;

		const isWhiteTurn = game.turn() === "w";
		const aiShouldMove =
			(gameMode === "ai-black" && !isWhiteTurn) ||
			(gameMode === "ai-white" && isWhiteTurn);

		if (aiShouldMove && !isAiThinking && !isFetching.current) {
			fetchAiMove(game);
		}
	}, [game, isGameOver, gameMode, isAiThinking, fetchAiMove]);

	const onDrop = useCallback(
		(args: {
			sourceSquare: string;
			targetSquare: string | null;
		}): boolean => {
			const { sourceSquare, targetSquare } = args;
			if (!targetSquare) return false;

			const isHumanTurn =
				gameMode === "pvp" ||
				(gameMode === "ai-black" && isWhiteTurn) ||
				(gameMode === "ai-white" && !isWhiteTurn);

			if (!isHumanTurn || isAiThinking) return false;

			const boardPiece = game.get(sourceSquare as Square);
			const isPromotion =
				boardPiece &&
				boardPiece.type === "p" &&
				(targetSquare[1] === "8" || targetSquare[1] === "1");

			const moveParams: { from: string; to: string; promotion?: string } =
				{
					from: sourceSquare,
					to: targetSquare,
				};
			if (isPromotion) moveParams.promotion = "q";

			const newGame = makeMove(
				{ from: sourceSquare, to: targetSquare },
				game,
			);
			return newGame !== null;
		},
		[game, gameMode, isWhiteTurn, isAiThinking, makeMove],
	);

	const startNewGame = useCallback((mode: GameMode) => {
		setGameMode(mode);
		setGame(new Chess());
		setAiScore(null);
		setIsAiThinking(false);
		fetchLock.current = false;
	}, []);

	const undoMove = useCallback(() => {
		if (isAiThinking) return;

		const gameCopy = new Chess();
		gameCopy.loadPgn(game.pgn());

		gameCopy.undo();
		if (gameMode !== "pvp") {
			gameCopy.undo();
		}

		setGame(gameCopy);
		setAiScore(null);
	}, [game, gameMode, isAiThinking]);

	const boardOrientation = gameMode === "ai-white" ? "black" : "white";

	const clampedScore = Math.max(-20, Math.min(20, aiScore || 0));
	const whitePercentage = 50 + (clampedScore / 20) * 50;

	return (
		<main className="flex flex-col justify-center items-center bg-gray-900 p-4 min-h-screen">
			<div className="flex flex-col items-center bg-gray-800 shadow-2xl p-8 rounded-2xl w-full max-w-2xl">
				<h1 className="mb-6 font-bold text-white text-3xl">
					Chess AI vs Human
				</h1>

				<div className="flex gap-2 mb-6">
					<button
						onClick={() => startNewGame("ai-black")}
						className={`px-4 py-2 rounded-lg font-semibold transition-colors ${gameMode === "ai-black" ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
					>
						Graj Białymi
					</button>
					<button
						onClick={() => startNewGame("ai-white")}
						className={`px-4 py-2 rounded-lg font-semibold transition-colors ${gameMode === "ai-white" ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
					>
						Graj Czarnymi
					</button>
					<button
						onClick={() => startNewGame("pvp")}
						className={`px-4 py-2 rounded-lg font-semibold transition-colors ${gameMode === "pvp" ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
					>
						PvP
					</button>
				</div>

				<div className="mb-4 w-full max-w-125">
					<div className="flex justify-between mb-2 px-1 text-gray-300">
						<div className="flex items-center gap-2">
							{isAiThinking ? (
								<span className="text-yellow-400 animate-pulse">
									🤖 AI myśli...
								</span>
							) : gameMode !== "pvp" ? (
								<span>🤖 AI czeka na twój ruch</span>
							) : (
								<span>👤 Gra lokalna</span>
							)}
						</div>
						{aiScore !== null && gameMode !== "pvp" && (
							<div className="font-mono font-bold text-sm">
								{aiScore > 0 ? "+" : ""}
								{aiScore.toFixed(2)}
							</div>
						)}
					</div>

					{/* ВИЗУАЛЬНАЯ ШКАЛА ОЦЕНКИ (EVAL BAR) */}
					<div className="flex bg-gray-900 border border-gray-700 rounded-full w-full h-3 overflow-hidden">
						<div
							className="bg-gray-200 h-full transition-all duration-700 ease-in-out"
							style={{ width: `${whitePercentage}%` }}
						/>
					</div>
				</div>

				<div className="shadow-lg rounded-md w-full max-w-125 overflow-hidden">
					<Chessboard
						options={{
							id: "BasicBoard",
							position: currentFen,
							onPieceDrop: onDrop,
							boardOrientation: boardOrientation,
						}}
					/>
				</div>

				{/* КНОПКИ УПРАВЛЕНИЯ */}
				<div className="flex gap-4 mt-8">
					<button
						className="bg-gray-600 hover:bg-gray-500 disabled:opacity-50 px-6 py-2 rounded-lg font-semibold text-white transition-colors"
						onClick={undoMove}
						disabled={game.history().length === 0 || isAiThinking}
					>
						Cofnij ruch
					</button>
					<button
						className="bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded-lg font-semibold text-white transition-colors"
						onClick={() => startNewGame(gameMode)}
					>
						Nowa gra
					</button>
				</div>
			</div>
		</main>
	);
}
