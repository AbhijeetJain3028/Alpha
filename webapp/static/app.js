/* INDUS web client */
const $ = (id) => document.getElementById(id);
const state = {
  sid: localStorage.getItem("indus.sid") || crypto.randomUUID(),
  convs: JSON.parse(localStorage.getItem("indus.convs") || "[]"),
  current: localStorage.getItem("indus.current") || null,
  temp: 0.7, topk: 50, maxtok: 256, system: "",
  busy: false,
};
localStorage.setItem("indus.sid", state.sid);

/* ───────────────────────── sidebar conversations ───────────────────── */
function persistConvs() {
  localStorage.setItem("indus.convs", JSON.stringify(state.convs));
  localStorage.setItem("indus.current", state.current ?? "");
}
function renderConvList() {
  const list = $("conv-list");
  list.innerHTML = "";
  for (const cv of [...state.convs].sort((a, b) => b.ts - a.ts).slice(0, 30)) {
    const btn = document.createElement("button");
    btn.className = "conv-item" + (cv.id === state.current ? " active" : "");
    btn.textContent = cv.title;
    btn.onclick = () => openConv(cv.id);
    list.appendChild(btn);
  }
}
function openConv(id) {
  const cv = state.convs.find(c => c.id === id);
  if (!cv) return;
  // switching conversations starts a fresh server session keyed per conv
  state.current = id; persistConvs();
  $("messages").innerHTML = "";
  for (const m of cv.msgs) addMessage(m.role, m.text, false);
  $("conv-title").textContent = cv.title;
  renderConvList();
}
function currentConv() {
  let cv = state.convs.find(c => c.id === state.current);
  if (!cv) {
    cv = { id: crypto.randomUUID(), title: "New conversation",
           ts: Date.now(), msgs: [] };
    state.convs.push(cv);
    state.current = cv.id;
    $("conv-title").textContent = cv.title;
  }
  return cv;
}

function addMessage(role, text, animate = true) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const av = document.createElement("div");
  av.className = "avatar";
  av.textContent = role === "you" ? "You"[0] : "I";
  const body = document.createElement("div");
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "you" ? "you" : "indus";
  const txt = document.createElement("div");
  txt.className = "text";
  txt.textContent = text;
  if (animate && role === "indus") {
    const cur = document.createElement("span");
    cur.className = "cursor";
    txt.appendChild(cur);
    txt._cursor = cur;
  }
  body.append(who, txt);
  if (role === "indus") body.appendChild(feedbackBar(text));
  wrap.append(av, body);
  $("welcome")?.style.setProperty("display", "none");
  $("messages").appendChild(wrap);
  scrollDown();
  return txt;
}

/* thumbs + teach-a-better-answer bar under each indus reply */
function feedbackBar(replyText) {
  const bar = document.createElement("div");
  bar.className = "feedback-bar";
  const up = document.createElement("button");
  up.className = "fb-btn"; up.title = "Good reply"; up.textContent = "👍";
  const down = document.createElement("button");
  down.className = "fb-btn"; down.title = "Bad reply"; down.textContent = "👎";
  const teach = document.createElement("textarea");
  teach.className = "teach-box";
  teach.placeholder = "Teach Indus a better answer… (⏎ to submit)";
  teach.hidden = true;
  const status = document.createElement("span");
  status.className = "fb-status";

  async function post(verdict, correction) {
    try {
      const r = await fetch("/api/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: state.sid, prompt: lastUserText(), reply: replyText,
          verdict, correction: correction || null,
        }),
      });
      status.textContent = r.ok ? "saved ✓ trains later" : "failed";
    } catch { status.textContent = "offline"; }
  }

  up.onclick = () => { up.disabled = down.disabled = true; post("up"); };
  down.onclick = () => {
    up.disabled = true; down.disabled = true;
    teach.hidden = false; teach.focus(); post("down");
  };
  teach.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const v = teach.value.trim();
      if (!v) return;
      teach.disabled = true; teach.hidden = false;
      post("taught", v);          // correction replaces the reply in training
      status.textContent = "taught ✓ trains later";
    }
  });
  bar.append(up, down, teach, status);
  return bar;
}
function lastUserText() {
  const msgs = [...document.querySelectorAll(".msg.you .text")];
  return msgs.length ? msgs[msgs.length - 1].textContent : "";
}
function scrollDown() {
  const sc = $("chat-scroll");
  sc.scrollTop = sc.scrollHeight;
}

/* ───────────────────────── chat streaming ─────────────────────── */
async function send(text) {
  if (state.busy || !text.trim()) return;
  state.busy = true;
  $("send").disabled = true;

  const cv = currentConv();
  if (cv.msgs.length === 0) {
    cv.title = text.slice(0, 42) + (text.length > 42 ? "…" : "");
    $("conv-title").textContent = cv.title;
    renderConvList(); persistConvs();
  }
  cv.msgs.push({ role: "you", text });
  addMessage("you", text);
  const target = addMessage("indus", "");

  let acc = "";
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sid + ":" + cv.id,
        message: text,
        temperature: state.temp,
        top_k: state.topk,
        max_tokens: state.maxtok,
        system: state.system,
      }),
    });
    if (!res.ok || !res.body) throw new Error(await res.text());
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const raw = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        if (!raw.startsWith("data:")) continue;
        const ev = JSON.parse(raw.slice(5));
        if (ev.t === "tok") {
          acc += ev.v;
          target.textContent = acc;
          target.appendChild(target._cursor);
          scrollDown();
        } else if (ev.t === "done") {
          $("rate").textContent =
            `${ev.tokens} tokens · ${ev.rate} tok/s`;
        }
      }
    }
  } catch (err) {
    target.textContent = acc || "(connection error — is the server up?)";
    console.error(err);
  }
  if (target._cursor) target._cursor.remove();
  cv.msgs.push({ role: "indus", text: acc });
  persistConvs();
  state.busy = false;
  $("send").disabled = false;
  $("input").focus();
}

/* ───────────────────────── settings drawer ────────────────────── */
function openDrawer() { $("drawer").hidden = false; $("drawer-backdrop").hidden = false; }
function closeDrawer() { $("drawer").hidden = true; $("drawer-backdrop").hidden = true; }

async function refreshInfo() {
  try {
    const r = await fetch("/api/info");
    const d = await r.json();
    $("status-dot").classList.add("ok");
    $("model-badge").innerHTML =
      `<b>${d.name}</b> · ${d.params_M}M params<br>` +
      `${d.checkpoint}<br>ctx ${d.block_size} · vocab ${d.vocab_size}` +
      `<br>${d.device.toUpperCase()}${d.chat_ready ? "" : "<br><i>base model (no chat tuning yet)</i>"}`;
    $("ctx-note").textContent = `ctx ${d.block_size}`;
  } catch {
    $("status-dot").classList.remove("ok");
    $("model-badge").textContent = "server offline";
  }
}

$("settings-btn").onclick = openDrawer;
$("close-drawer").onclick = closeDrawer;
$("drawer-backdrop").onclick = closeDrawer;
$("set-temp").oninput = e => { state.temp = +e.target.value; $("temp-out").value = state.temp.toFixed(2); };
$("set-topk").oninput = e => { state.topk = +e.target.value; $("topk-out").value = state.topk; };
$("set-maxtok").oninput = e => { state.maxtok = +e.target.value; $("maxtok-out").value = state.maxtok; };
$("set-system").oninput = e => { state.system = e.target.value; };
$("load-ckpt").onclick = async () => {
  $("load-status").textContent = "loading…";
  try {
    const r = await fetch("/api/load", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ which: $("set-ckpt").value.trim() }),
    });
    const d = await r.json();
    $("load-status").textContent = d.ok ? "loaded ✓" : d.detail;
    refreshInfo();
  } catch (e) { $("load-status").textContent = String(e); }
};

/* ───────────────────────── composer wiring ────────────────────── */
$("new-chat").onclick = () => {
  state.current = null;
  $("messages").innerHTML = "";
  $("welcome").style.display = "";
  $("conv-title").textContent = "New conversation";
  renderConvList(); persistConvs();
};
document.querySelectorAll(".starter").forEach(b =>
  b.onclick = () => send(b.textContent));
$("send").onclick = () => { const v = $("input").value; $("input").value = ""; send(v); };
$("input").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    const v = $("input").value; $("input").value = ""; autoGrow();
    send(v);
  }
});
function autoGrow() {
  const el = $("input");
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 180) + "px";
}
$("input").addEventListener("input", autoGrow);

renderConvList();
refreshInfo();
setInterval(refreshInfo, 30000);
