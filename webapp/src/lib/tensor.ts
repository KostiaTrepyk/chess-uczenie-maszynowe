import * as ort from "onnxruntime-node";

const PIECE_TO_CHANNEL: Record<string, number> = {
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

export function createBatchTensor(fens: string[]): ort.Tensor {
	const batchSize = fens.length;
	const data = new Float32Array(batchSize * 15 * 64);

	for (let b = 0; b < batchSize; b++) {
		const parts = fens[b].split(" ");
		const board = parts[0];
		const isBlackTurn = parts[1] === "b";
		const offset = b * 960; // 15 каналов * 64 клетки

		// --- 1. ПАРСИНГ ФИГУР (С КАНОНИЧЕСКИМ ОТРАЖЕНИЕМ) ---
		let row = 0;
		let col = 0;

		for (const char of board) {
			if (char === "/") {
				row++;
				col = 0;
			} else if (/\d/.test(char)) {
				col += parseInt(char, 10);
			} else {
				let channel = PIECE_TO_CHANNEL[char];
				let actualRow = row;

				// МАГИЯ ИЗ features.py: Если ход черных, переворачиваем доску и меняем цвета
				if (isBlackTurn) {
					actualRow = 7 - row; // Отражение по вертикали
					channel = channel < 6 ? channel + 6 : channel - 6; // Свап цветов
				}

				if (channel !== undefined) {
					data[offset + channel * 64 + actualRow * 8 + col] = 1.0;
				}
				col++;
			}
		}

		// --- 2. ОЧЕРЕДЬ ХОДА ---
		// В твоем features.py это просто слой из 1.0, так как сеть всегда "ходит своими"
		data.fill(1.0, offset + 12 * 64, offset + 13 * 64);

		// --- 3. РОКИРОВКА ---
		const cStr = parts[2] || "-";
		const K = cStr.includes("K") ? 1.0 : 0.0;
		const Q = cStr.includes("Q") ? 1.0 : 0.0;
		const k = cStr.includes("k") ? 1.0 : 0.0;
		const q = cStr.includes("q") ? 1.0 : 0.0;

		const ourShort = isBlackTurn ? k : K;
		const ourLong = isBlackTurn ? q : Q;
		const oppShort = isBlackTurn ? K : k;
		const oppLong = isBlackTurn ? Q : q;

		const castlingOffset = offset + 13 * 64;
		data[castlingOffset + 7 * 8 + 7] = ourShort;
		data[castlingOffset + 7 * 8 + 0] = ourLong;
		data[castlingOffset + 0 * 8 + 7] = oppShort;
		data[castlingOffset + 0 * 8 + 0] = oppLong;

		// --- 4. ВЗЯТИЕ НА ПРОХОДЕ (EN PASSANT) ---
		if (parts[3] !== "-") {
			const epCol = parts[3].charCodeAt(0) - 97; // 'a' это 97
			let epRow = 8 - parseInt(parts[3][1], 10);

			if (isBlackTurn) epRow = 7 - epRow; // Отражение по вертикали (^ 56 в питоне)

			const epOffset = offset + 14 * 64;
			data[epOffset + epRow * 8 + epCol] = 1.0;
		}
	}

	return new ort.Tensor("float32", data, [batchSize, 15, 8, 8]);
}

export function decodeEvaluations(
	logits: Float32Array,
	batchSize: number,
): number[] {
	const scores: number[] = [];
	const WDL_SCALE = 3.0; // Берем из твоего consts.py!

	for (let b = 0; b < batchSize; b++) {
		const rawLogit = logits[b];

		if (rawLogit === undefined || Number.isNaN(rawLogit)) {
			scores.push(0);
			continue;
		}

		let p = 1.0 / (1.0 + Math.exp(-rawLogit));
		p = Math.max(1e-4, Math.min(1.0 - 1e-4, p));

		const pawns = -WDL_SCALE * Math.log(1.0 / p - 1.0);

		scores.push(pawns);
	}

	return scores;
}
