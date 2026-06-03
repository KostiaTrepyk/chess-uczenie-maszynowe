import * as ort from "onnxruntime-node";
import path from "path";

let session: ort.InferenceSession | null = null;

export async function getModelSession(): Promise<ort.InferenceSession> {
	if (!session) {
		// Указываем путь к папке models в корне проекта
		const modelPath = path.join(
			process.cwd(),
			"models",
			"chess_model.onnx",
		);
		session = await ort.InferenceSession.create(modelPath, {
			executionProviders: ["cpu"], // Для сервера Next.js на начальном этапе используем CPU
		});
	}
	return session;
}
