"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Chess, Square } from "chess.js";
import { Chessboard as OriginalChessboard } from "react-chessboard";

const Chessboard = OriginalChessboard;

type GameMode = "ai-black" | "ai-white" | "pvp" | "model-vs-model";

type MoveInput = Parameters<Chess["move"]>[0];

export default function Home() {
	const [game, setGame] = useState(new Chess());
	const [gameMode, setGameMode] = useState<GameMode>("ai-black");
	const [isAiThinking, setIsAiThinking] = useState(false);
	const [aiScore, setAiScore] = useState<number | null>(null);
	const [isModelVsModelPaused, setIsModelVsModelPaused] = useState(false);
	const [moveDelay, setMoveDelay] = useState<number>(1); // seconds for model-vs-model
	const [searchMode, setSearchMode] = useState<"simple" | "advanced">(
		"simple",
	);
	const [searchDepth, setSearchDepth] = useState<number>(2);

	const isFetching = useRef(false);
	const modelMoveTimerRef = useRef<number | null>(null);

	const currentFen = game.fen();
	const isWhiteTurn = game.turn() === "w";
	const isGameOver = game.isGameOver();

	const makeMove = useCallback((move: MoveInput, currentGame: Chess) => {
		try {
			const gameCopy = new Chess();
			gameCopy.loadPgn(currentGame.pgn());
			const result = gameCopy.move(move);
			if (result) {
				setGame(gameCopy);
				return gameCopy;
			}
		} catch {
			return null;
		}
		return null;
	}, []);

	const fetchAiMove = useCallback(
		async (currentGame: Chess) => {
			if (isFetching.current) return;
			isFetching.current = true;
			setIsAiThinking(true);

			try {
				const response = await fetch("/api/move", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						fen: currentGame.fen(),
						history: currentGame.history(),
						searchMode,
						searchDepth,
					}),
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
		[makeMove, searchMode, searchDepth],
	);

	useEffect(() => {
		if (modelMoveTimerRef.current) {
			clearTimeout(modelMoveTimerRef.current);
			modelMoveTimerRef.current = null;
		}

		if (isGameOver || isModelVsModelPaused) return;

		const isWhiteTurn = game.turn() === "w";
		const aiShouldMove =
			gameMode === "model-vs-model" ||
			(gameMode === "ai-black" && !isWhiteTurn) ||
			(gameMode === "ai-white" && isWhiteTurn);

		if (!aiShouldMove || isAiThinking || isFetching.current) return;

		if (gameMode === "model-vs-model") {
			modelMoveTimerRef.current = window.setTimeout(
				() => {
					modelMoveTimerRef.current = null;
					fetchAiMove(game);
				},
				Math.max(0, moveDelay) * 1000,
			);
			return;
		}

		modelMoveTimerRef.current = window.setTimeout(() => {
			modelMoveTimerRef.current = null;
			fetchAiMove(game);
		}, 0);
	}, [
		game,
		isGameOver,
		gameMode,
		isAiThinking,
		isModelVsModelPaused,
		moveDelay,
		fetchAiMove,
	]);

	useEffect(() => {
		return () => {
			if (modelMoveTimerRef.current) {
				clearTimeout(modelMoveTimerRef.current);
				modelMoveTimerRef.current = null;
			}
		};
	}, []);

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

			const newGame = makeMove(moveParams, game);
			return newGame !== null;
		},
		[game, gameMode, isWhiteTurn, isAiThinking, makeMove],
	);

	const startNewGame = useCallback((mode: GameMode) => {
		setGameMode(mode);
		setGame(new Chess());
		setAiScore(null);
		setIsAiThinking(false);
		setIsModelVsModelPaused(false);
	}, []);

	const toggleModelVsModelPause = useCallback(() => {
		if (gameMode !== "model-vs-model") return;

		setIsModelVsModelPaused((paused) => !paused);
	}, [gameMode]);

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
					<button
						onClick={() => startNewGame("model-vs-model")}
						className={`px-4 py-2 rounded-lg font-semibold transition-colors ${gameMode === "model-vs-model" ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
					>
						Model vs Model
					</button>
				</div>

				<div className="mb-4 w-full max-w-125">
					<div className="flex justify-between mb-2 px-1 text-gray-300">
						<div className="flex items-center gap-2">
							{isAiThinking ? (
								<span className="text-yellow-400 animate-pulse">
									🤖 AI myśli...
								</span>
							) : gameMode === "model-vs-model" ? (
								<span>🤖 Modele grają między sobą</span>
							) : gameMode !== "pvp" ? (
								<span>🤖 AI czeka na twój ruch</span>
							) : (
								<span>👤 Gra lokalna</span>
							)}
						</div>
						{aiScore !== null && gameMode !== "pvp" && (
							<div className="font-mono font-bold text-sm whitespace-nowrap">
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
						disabled={
							game.history().length === 0 ||
							isAiThinking ||
							gameMode === "model-vs-model"
						}
					>
						Cofnij ruch
					</button>
					{gameMode === "model-vs-model" && (
						<button
							className={`px-6 py-2 rounded-lg font-semibold text-white transition-colors ${isModelVsModelPaused ? "bg-emerald-600 hover:bg-emerald-500" : "bg-amber-600 hover:bg-amber-500"}`}
							onClick={toggleModelVsModelPause}
							disabled={isGameOver}
						>
							{isModelVsModelPaused ? "Kontynuuj" : "Pauza"}
						</button>
					)}
					{gameMode === "model-vs-model" && (
						<div className="flex items-center gap-3 px-2">
							<label className="text-gray-300 text-sm">
								Interwał:
							</label>
							<input
								type="range"
								min={0}
								max={5}
								step={0.1}
								value={moveDelay}
								onChange={(e) =>
									setMoveDelay(
										parseFloat(
											(e.target as HTMLInputElement)
												.value,
										),
									)
								}
								className="w-40"
							/>
							<div className="w-12 text-gray-200 text-sm">
								{moveDelay.toFixed(1)}s
							</div>
						</div>
					)}
					<button
						className="bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded-lg font-semibold text-white transition-colors"
						onClick={() => startNewGame(gameMode)}
					>
						Nowa gra
					</button>
				</div>
			</div>
			<div className="flex items-center gap-4 mt-4">
				<div className="text-gray-300 text-sm">Tryb ruchu:</div>
				<button
					className={`px-3 py-1 rounded-md text-sm ${searchMode === "simple" ? "bg-blue-700 text-gray-200" : "bg-gray-700 text-white"}`}
					onClick={() => setSearchMode("simple")}
				>
					Prosty
				</button>
				<button
					className={`px-3 py-1 rounded-md text-sm ${searchMode === "advanced" ? "bg-blue-700 text-gray-200" : "bg-gray-700 text-white"}`}
					onClick={() => setSearchMode("advanced")}
				>
					Zaawansowany
				</button>
				{searchMode === "advanced" && (
					<div className="flex items-center gap-2">
						<label className="text-gray-300 text-sm">
							Głębokość:
						</label>
						<input
							type="number"
							min={1}
							max={5}
							value={searchDepth}
							onChange={(e) =>
								setSearchDepth(
									Math.max(
										1,
										Math.min(
											6,
											parseInt(
												(e.target as HTMLInputElement)
													.value || "1",
											),
										),
									),
								)
							}
							className="bg-gray-700 px-2 py-1 rounded-md w-16 text-white text-sm"
						/>
					</div>
				)}
			</div>
		</main>
	);
}
