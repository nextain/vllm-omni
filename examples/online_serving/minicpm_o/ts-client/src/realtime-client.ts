/**
 * Reference TypeScript client for vllm-omni /v1/realtime (MiniCPM-o 4.5).
 *
 * Sibling of `realtime_e2e_test.py`. Same wire protocol, same single-turn
 * scope (no conversation history accumulation — the server intentionally
 * disables multi-turn history; see `.agents/context/contribution-journey.md`
 * Phase 11 lesson 16).
 *
 * Phase 11 server constraints honored here:
 *   - `server_vad` is declared-but-unimplemented; we drive responses with
 *     explicit `input_audio_buffer.commit` (Phase 11 lesson 15).
 *   - Single turn per WebSocket. No history reuse.
 *
 * Implementation note — single message handler:
 *   The session has two phases: a handshake phase that resolves once
 *   `session.created` arrives, and a response phase that resolves once
 *   `response.done` arrives. Both phases share ONE `ws.on("message")`
 *   listener that branches on a `stage` enum. This avoids the double-bind
 *   pattern (one listener for connect, a second for response) that would
 *   otherwise double-count audio bytes and transcript deltas during the
 *   overlap window.
 */
import WebSocket, { type RawData } from "ws";

/** Schemes accepted for `serverUrl` before conversion to ws(s)://. */
const ALLOWED_SERVER_SCHEMES = new Set(["http:", "https:", "ws:", "wss:"]);

const DEFAULT_MODEL = "openbmb/MiniCPM-o-4_5";

export interface ClientConfig {
	/** http(s):// or ws(s):// origin. Path is appended automatically. */
	serverUrl: string;
	/** Model id (default: openbmb/MiniCPM-o-4_5). */
	model?: string;
	/** System instruction passed via `session.update.session.instructions`. */
	instructions?: string;
	/** Sampling temperature for the response. */
	temperature?: number;
	/** Connection timeout in ms (default: 15000). */
	connectTimeoutMs?: number;
	/** Per-event timeout for the response stream (default: 30000). */
	responseTimeoutMs?: number;
}

export interface ResponseSummary {
	receivedSessionCreated: boolean;
	receivedResponseCreated: boolean;
	receivedAudio: boolean;
	receivedTranscript: boolean;
	receivedDone: boolean;
	transcript: string;
	audioChunks: number;
	audioBytes: number;
	timeline: Array<{ event: string; tSec: number }>;
}

type Stage = "connecting" | "responding" | "done";

/**
 * Run a single-turn realtime exchange:
 *   connect → wait session.created → session.update → append PCM16 chunks
 *   → commit → collect response stream → close.
 *
 * `pcm16Chunks` MUST be 16 kHz mono PCM16 little-endian, already chunked
 * (e.g., 1-second slices). Each chunk is base64-encoded and sent as
 * `input_audio_buffer.append`.
 */
export async function runRealtimeTurn(
	cfg: ClientConfig,
	pcm16Chunks: Buffer[],
): Promise<ResponseSummary> {
	const normalizedBase = normalizeServerUrl(cfg.serverUrl);
	if (!normalizedBase) {
		throw new Error(
			`Invalid serverUrl: expected http(s):// or ws(s)://, got ${cfg.serverUrl}`,
		);
	}
	const wsUrl = `${normalizedBase}/v1/realtime`;
	const connectTimeoutMs = cfg.connectTimeoutMs ?? 15000;
	const responseTimeoutMs = cfg.responseTimeoutMs ?? 30000;

	console.log(`Connecting to ${sanitizeUrl(wsUrl)}...`);

	const summary: ResponseSummary = {
		receivedSessionCreated: false,
		receivedResponseCreated: false,
		receivedAudio: false,
		receivedTranscript: false,
		receivedDone: false,
		transcript: "",
		audioChunks: 0,
		audioBytes: 0,
		timeline: [],
	};

	const t0 = process.hrtime.bigint();
	const elapsed = (): number =>
		Number(process.hrtime.bigint() - t0) / 1_000_000_000;

	let stage: Stage = "connecting";
	let resolveConnect!: () => void;
	let rejectConnect!: (err: Error) => void;
	let resolveResponse!: () => void;
	let rejectResponse!: (err: Error) => void;

	const connectPromise = new Promise<void>((res, rej) => {
		resolveConnect = res;
		rejectConnect = rej;
	});
	const responsePromise = new Promise<void>((res, rej) => {
		resolveResponse = res;
		rejectResponse = rej;
	});

	const ws = new WebSocket(wsUrl);

	const connectTimer = setTimeout(() => {
		if (stage !== "connecting") return;
		stage = "done";
		rejectConnect(new Error("Connection timeout"));
		try {
			ws.close();
		} catch {
			// already closed
		}
	}, connectTimeoutMs);

	let responseTimer: NodeJS.Timeout | null = null;

	ws.on("message", (data: RawData) => {
		const msg = parseMessage(data);
		if (!msg) return;
		const type = msg.type;

		if (stage === "connecting") {
			if (type === "session.created") {
				clearTimeout(connectTimer);
				summary.receivedSessionCreated = true;
				summary.timeline.push({
					event: "session.created",
					tSec: elapsed(),
				});
				ws.send(
					JSON.stringify({
						// Fork extension: top-level `model`. The omni handler does
						// not actually consume this field today, but the Python
						// sibling sends it for forward-compat.
						type: "session.update",
						model: cfg.model ?? DEFAULT_MODEL,
						session: {
							instructions: cfg.instructions ?? "",
							temperature: cfg.temperature ?? 0.6,
						},
					}),
				);
				stage = "responding";
				resolveConnect();
				return;
			}
			if (type === "error") {
				clearTimeout(connectTimer);
				stage = "done";
				rejectConnect(new Error(extractServerErrorMessage(msg)));
				return;
			}
			// Unexpected event during handshake — ignore.
			return;
		}

		if (stage === "responding") {
			if (type === "error") {
				if (responseTimer) clearTimeout(responseTimer);
				stage = "done";
				rejectResponse(new Error(extractServerErrorMessage(msg)));
				return;
			}
			handleResponseEvent(msg, summary, elapsed);
			if (type === "response.done") {
				if (responseTimer) clearTimeout(responseTimer);
				stage = "done";
				resolveResponse();
			}
			return;
		}

		// stage === "done": discard late frames.
	});

	ws.on("error", (err) => {
		clearTimeout(connectTimer);
		if (responseTimer) clearTimeout(responseTimer);
		if (stage === "connecting") rejectConnect(err);
		else if (stage === "responding") rejectResponse(err);
		stage = "done";
	});

	ws.on("close", () => {
		clearTimeout(connectTimer);
		if (responseTimer) clearTimeout(responseTimer);
		if (stage === "connecting") {
			rejectConnect(new Error("Connection closed before session.created"));
		} else if (stage === "responding") {
			rejectResponse(new Error("Connection closed before response.done"));
		}
		stage = "done";
	});

	await connectPromise;

	const totalBytes = pcm16Chunks.reduce((acc, c) => acc + c.length, 0);
	console.log(
		`Sending ${pcm16Chunks.length} audio chunks (${totalBytes} bytes total)...`,
	);
	for (const chunk of pcm16Chunks) {
		ws.send(
			JSON.stringify({
				type: "input_audio_buffer.append",
				audio: chunk.toString("base64"),
			}),
		);
	}
	ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
	summary.timeline.push({
		event: "input_audio_buffer.commit",
		tSec: elapsed(),
	});
	console.log("Committed. Waiting for response stream...");

	responseTimer = setTimeout(() => {
		if (stage !== "responding") return;
		stage = "done";
		rejectResponse(
			new Error(`No response.done within ${responseTimeoutMs / 1000}s`),
		);
	}, responseTimeoutMs);

	try {
		await responsePromise;
	} finally {
		if (responseTimer) clearTimeout(responseTimer);
		try {
			ws.close();
		} catch {
			// already closed
		}
	}

	return summary;
}

function parseMessage(data: RawData): Record<string, unknown> | null {
	try {
		const text = typeof data === "string" ? data : data.toString();
		const obj = JSON.parse(text);
		if (!obj || typeof obj !== "object" || Array.isArray(obj)) {
			return null;
		}
		return obj as Record<string, unknown>;
	} catch {
		return null;
	}
}

function handleResponseEvent(
	msg: Record<string, unknown>,
	summary: ResponseSummary,
	elapsed: () => number,
): void {
	const type = msg.type;
	if (typeof type !== "string") return;
	switch (type) {
		case "response.created":
			if (!summary.receivedResponseCreated) {
				summary.receivedResponseCreated = true;
				summary.timeline.push({ event: type, tSec: elapsed() });
			}
			break;
		case "response.audio_transcript.delta": {
			const delta = msg.delta;
			if (typeof delta === "string") {
				summary.transcript += delta;
				if (!summary.receivedTranscript) {
					summary.timeline.push({
						event: "response.audio_transcript.delta:first",
						tSec: elapsed(),
					});
				}
				summary.receivedTranscript = true;
			}
			break;
		}
		case "response.audio.delta": {
			const delta = msg.delta;
			if (typeof delta === "string") {
				summary.audioChunks++;
				summary.audioBytes += Buffer.byteLength(delta, "base64");
				if (summary.audioChunks === 1) {
					summary.timeline.push({
						event: "response.audio.delta:first",
						tSec: elapsed(),
					});
				}
				summary.receivedAudio = true;
			}
			break;
		}
		case "response.audio.done":
		case "response.audio_transcript.done":
			summary.timeline.push({ event: type, tSec: elapsed() });
			break;
		case "response.done":
			summary.receivedDone = true;
			summary.timeline.push({ event: type, tSec: elapsed() });
			break;
	}
}

/**
 * Validate and normalize a server URL to a trailing-slash-stripped
 * ws(s):// origin. Returns null for invalid input.
 *
 * Accepts http(s):// and ws(s):// with a scheme allowlist so `HTTP://` or
 * `ftp://...` fail fast in `runRealtimeTurn` rather than as an opaque throw
 * inside `new WebSocket()`. Embedded credentials are stripped — vllm-omni
 * has no auth; userinfo in the URL is always a misconfiguration.
 */
function normalizeServerUrl(raw: string): string | null {
	let parsed: URL;
	try {
		parsed = new URL(raw);
	} catch {
		return null;
	}
	const scheme = parsed.protocol.toLowerCase();
	if (!ALLOWED_SERVER_SCHEMES.has(scheme)) return null;
	parsed.protocol = scheme.startsWith("http")
		? scheme.replace("http", "ws")
		: scheme;
	parsed.username = "";
	parsed.password = "";
	const origin = `${parsed.protocol}//${parsed.host}`;
	return parsed.pathname && parsed.pathname !== "/"
		? `${origin}${parsed.pathname.replace(/\/+$/, "")}`
		: origin;
}

/** Strip userinfo from a URL string before logging. */
function sanitizeUrl(url: string): string {
	try {
		const u = new URL(url);
		if (u.username || u.password) {
			u.username = "";
			u.password = "";
			return u.toString();
		}
		return url;
	} catch {
		return url;
	}
}

function extractServerErrorMessage(msg: Record<string, unknown>): string {
	const err = msg.error;
	if (typeof err === "string") return err;
	if (err && typeof err === "object") {
		const m = (err as Record<string, unknown>).message;
		if (typeof m === "string") return m;
	}
	const top = msg.message;
	if (typeof top === "string") return top;
	return "Server error";
}
