import * as ort from "onnxruntime-node";
import path from "path";

let session: ort.InferenceSession | null = null;

export async function getModelSession(): Promise<ort.InferenceSession> {
	if (!session) {
		const modelPath = path.join(process.cwd(), "models", "chess_model.onnx");
		
		session = await ort.InferenceSession.create(modelPath, { 
			executionProviders: ["cpu"],
			graphOptimizationLevel: 'all', // Максимальная оптимизация графа
			intraOpNumThreads: 4,          // Распараллеливание внутри одной операции (поставь по числу ядер)
			executionMode: 'sequential'
		});
		
		console.log("✅ Sesja ONNX (Zoptymalizowana) uruchomiona na CPU.");
	}
	return session;
}