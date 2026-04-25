/**
 * E2E test for vllm-omni /v1/realtime with MiniCPM-o 4.5.
 *
 * TypeScript sibling of `realtime_e2e_test.py`. Reads a 16 kHz mono PCM16
 * WAV file, runs a single-turn realtime exchange, and asserts that the
 * server emitted both audio and transcript deltas. Exit code 0 on success,
 * 1 on failure.
 *
 * Usage:
 *   npm install
 *   npm run e2e -- --audio /path/to/16k_mono.wav --host localhost --port 8000
 */
import { readFileSync } from "node:fs";
import { argv, exit, hrtime } from "node:process";
import { runRealtimeTurn, type ResponseSummary } from "./realtime-client.js";

interface Args {
	audio: string;
	host: string;
	port: number;
	model: string;
}

function parseArgs(rawArgv: string[]): Args {
	const args: Partial<Args> = {
		host: "localhost",
		port: 8000,
		model: "openbmb/MiniCPM-o-4_5",
	};
	for (let i = 0; i < rawArgv.length; i++) {
		const a = rawArgv[i];
		switch (a) {
			case "--audio":
				args.audio = rawArgv[++i];
				break;
			case "--host":
				args.host = rawArgv[++i];
				break;
			case "--port": {
				const port = Number(rawArgv[++i]);
				if (!Number.isFinite(port) || port <= 0 || port > 65535) {
					throw new Error(`Invalid --port`);
				}
				args.port = port;
				break;
			}
			case "--model":
				args.model = rawArgv[++i];
				break;
			case "-h":
			case "--help":
				printUsage();
				exit(0);
				break;
			default:
				if (a?.startsWith("--")) {
					throw new Error(`Unknown flag: ${a}`);
				}
		}
	}
	if (!args.audio) {
		throw new Error("--audio <path> is required");
	}
	return args as Args;
}

function printUsage(): void {
	console.log(
		`Usage: npm run e2e -- --audio <path/to.wav> [--host localhost] [--port 8000] [--model openbmb/MiniCPM-o-4_5]`,
	);
}

/**
 * Read a 16 kHz mono PCM16 WAV file and split into 1-second chunks.
 *
 * Validates only what we need:
 *   - RIFF/WAVE/fmt /data chunks present
 *   - audioFormat == 1 (PCM)
 *   - numChannels == 1
 *   - sampleRate == 16000
 *   - bitsPerSample == 16
 */
function loadWavAs1sChunks(path: string): Buffer[] {
	const buf = readFileSync(path);
	if (buf.length < 44) {
		throw new Error(`WAV too short: ${path}`);
	}
	if (buf.toString("ascii", 0, 4) !== "RIFF") {
		throw new Error(`Not a RIFF file: ${path}`);
	}
	if (buf.toString("ascii", 8, 12) !== "WAVE") {
		throw new Error(`Not a WAVE file: ${path}`);
	}

	let offset = 12;
	let audioFormat = -1;
	let numChannels = -1;
	let sampleRate = -1;
	let bitsPerSample = -1;
	let dataOffset = -1;
	let dataSize = -1;

	while (offset + 8 <= buf.length) {
		const chunkId = buf.toString("ascii", offset, offset + 4);
		const chunkSize = buf.readUInt32LE(offset + 4);
		const bodyStart = offset + 8;
		if (chunkId === "fmt ") {
			audioFormat = buf.readUInt16LE(bodyStart);
			numChannels = buf.readUInt16LE(bodyStart + 2);
			sampleRate = buf.readUInt32LE(bodyStart + 4);
			bitsPerSample = buf.readUInt16LE(bodyStart + 14);
		} else if (chunkId === "data") {
			dataOffset = bodyStart;
			dataSize = chunkSize;
			break;
		}
		// chunks are word-aligned
		offset = bodyStart + chunkSize + (chunkSize & 1);
	}

	if (dataOffset < 0) throw new Error(`No 'data' chunk in ${path}`);
	if (audioFormat !== 1) {
		throw new Error(
			`Unsupported audioFormat=${audioFormat} (expected PCM=1) in ${path}`,
		);
	}
	if (numChannels !== 1) {
		throw new Error(
			`Unsupported numChannels=${numChannels} (expected 1) in ${path}`,
		);
	}
	if (sampleRate !== 16000) {
		throw new Error(
			`Unsupported sampleRate=${sampleRate} (expected 16000) in ${path}`,
		);
	}
	if (bitsPerSample !== 16) {
		throw new Error(
			`Unsupported bitsPerSample=${bitsPerSample} (expected 16) in ${path}`,
		);
	}

	const dataEnd = Math.min(dataOffset + dataSize, buf.length);
	const pcm = buf.subarray(dataOffset, dataEnd);
	const bytesPerSecond = 16000 * 2; // 16kHz × 2 bytes
	const chunks: Buffer[] = [];
	for (let i = 0; i < pcm.length; i += bytesPerSecond) {
		chunks.push(pcm.subarray(i, Math.min(i + bytesPerSecond, pcm.length)));
	}
	return chunks;
}

function printSummary(summary: ResponseSummary): void {
	console.log("\nReceiving response events:");
	for (const entry of summary.timeline) {
		console.log(`  [+${entry.tSec.toFixed(2)}s] ${entry.event}`);
	}
	console.log("\nTest Summary:");
	console.log(`  Transcript received: ${summary.receivedTranscript}`);
	if (summary.receivedTranscript) {
		console.log(`  Final Transcript:    ${summary.transcript}`);
	}
	console.log(
		`  Audio received:      ${summary.receivedAudio} (${summary.audioChunks} chunks, ${summary.audioBytes} bytes)`,
	);
}

async function main(): Promise<void> {
	const t0 = hrtime.bigint();
	let args: Args;
	try {
		args = parseArgs(argv.slice(2));
	} catch (err) {
		console.error(`error: ${(err as Error).message}\n`);
		printUsage();
		exit(2);
	}

	const chunks = loadWavAs1sChunks(args.audio);
	const totalSamples = chunks.reduce((acc, c) => acc + c.length / 2, 0);
	console.log(
		`Loaded ${args.audio}: ${totalSamples} samples in ${chunks.length} chunks (1s each)`,
	);

	const summary = await runRealtimeTurn(
		{
			serverUrl: `http://${args.host}:${args.port}`,
			model: args.model,
		},
		chunks,
	);

	printSummary(summary);

	const ok =
		summary.receivedSessionCreated &&
		summary.receivedAudio &&
		summary.receivedTranscript &&
		summary.receivedDone;

	const totalSec = Number(hrtime.bigint() - t0) / 1_000_000_000;
	if (ok) {
		console.log(`\nRESULT: SUCCESS (${totalSec.toFixed(2)}s wall)`);
		exit(0);
	} else {
		console.log(`\nRESULT: FAILED (missing audio or transcript or done)`);
		exit(1);
	}
}

main().catch((err) => {
	console.error(`E2E error: ${err instanceof Error ? err.message : err}`);
	exit(1);
});
