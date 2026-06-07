import * as ort from "onnxruntime-node";

// 0-5: Свои фигуры, 6-11: Чужие фигуры
const PIECE_TO_CHANNEL_WHITE: Record<string, number> = {
	P: 0,
	N: 1,
	B: 2,
	R: 3,
	Q: 4,
	K: 5,
	p: 6,
	n: 7,
	b: 8,
	r: 9,
	q: 10,
	k: 11,
};

// Если ход черных, мы меняем их фигуры местами, делая черные "своими" (0-5)
const PIECE_TO_CHANNEL_BLACK: Record<string, number> = {
	p: 0,
	n: 1,
	b: 2,
	r: 3,
	q: 4,
	k: 5,
	P: 6,
	N: 7,
	B: 8,
	R: 9,
	Q: 10,
	K: 11,
};

export function createResNetTensor(fen: string) {
	// Создаем плоский массив для тензора 15 x 8 x 8 (960 элементов)
	const tensorData = new Float32Array(15 * 8 * 8).fill(0.0);
	const parts = fen.split(" ");
	const board = parts[0];
	const isWhiteTurn = parts[1] === "w";
	const castling = parts[2] || "-";
	const epStr = parts[3] || "-";

	// 1. СЛОИ 0-11: ФИГУРЫ
	let rank = 0; // 0 - верхняя строка доски (8-я горизонталь)
	let file = 0;

	for (const char of board) {
		if (char === "/") {
			rank++;
			file = 0;
		} else if (/\d/.test(char)) {
			file += parseInt(char, 10);
		} else {
			// КАНОНИЧЕСКАЯ МАГИЯ: Если ход черных, переворачиваем доску по вертикали
			const mappedRank = isWhiteTurn ? rank : 7 - rank;
			const sq = mappedRank * 8 + file;
			const channel = isWhiteTurn
				? PIECE_TO_CHANNEL_WHITE[char]
				: PIECE_TO_CHANNEL_BLACK[char];

			tensorData[channel * 64 + sq] = 1.0;
			file++;
		}
	}

	// 2. СЛОЙ 12: ОЧЕРЕДЬ ХОДА (Всегда заполняем единицами, как в Python)
	const ch12 = 12 * 64;
	for (let i = 0; i < 64; i++) {
		tensorData[ch12 + i] = 1.0;
	}

	// 3. СЛОЙ 13: РОКИРОВКИ
	const ch13 = 13 * 64;
	if (isWhiteTurn) {
		if (castling.includes("K")) tensorData[ch13 + 63] = 1.0; // Своя короткая (7,7)
		if (castling.includes("Q")) tensorData[ch13 + 56] = 1.0; // Своя длинная (7,0)
		if (castling.includes("k")) tensorData[ch13 + 7] = 1.0; // Чужая короткая (0,7)
		if (castling.includes("q")) tensorData[ch13 + 0] = 1.0; // Чужая длинная (0,0)
	} else {
		// При перевороте доски, рокировки тоже "меняются местами"
		if (castling.includes("k")) tensorData[ch13 + 63] = 1.0;
		if (castling.includes("q")) tensorData[ch13 + 56] = 1.0;
		if (castling.includes("K")) tensorData[ch13 + 7] = 1.0;
		if (castling.includes("Q")) tensorData[ch13 + 0] = 1.0;
	}

	// 4. СЛОЙ 14: ВЗЯТИЕ НА ПРОХОДЕ (En Passant)
	if (epStr !== "-") {
		const col = epStr.charCodeAt(0) - "a".charCodeAt(0);
		const fenRow = 8 - parseInt(epStr[1], 10); // 0-based сверху
		const mappedRow = isWhiteTurn ? fenRow : 7 - fenRow;

		tensorData[14 * 64 + mappedRow * 8 + col] = 1.0;
	}

	// Возвращаем объект с ключом 'input' (так мы назвали его при экспорте в ONNX)
	return {
		input: new ort.Tensor("float32", tensorData, [1, 15, 8, 8]),
	};
}
