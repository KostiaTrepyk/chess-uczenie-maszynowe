import * as ort from "onnxruntime-node";
import path from "path";

let session: ort.InferenceSession | null = null;

export async function getModelSession(): Promise<ort.InferenceSession> {
	if (!session) {
		const modelPath = path.join(process.cwd(), "models", "chess_model.onnx");
		
		session = await ort.InferenceSession.create(modelPath, { 
			executionProviders: ["cpu"],
			graphOptimizationLevel: 'all', 
			intraOpNumThreads: 0,  // <--- 0 = ИСПОЛЬЗОВАТЬ ВСЕ ЯДРА ПРОЦЕССОРА
			executionMode: 'sequential'
		});
		
		console.log("✅ Sesja ONNX uruchomiona na CPU (Max Threads).");
	}
	return session;
}