"""Fill the Hugging Face (Workable) forms. Stops before Submit unless :submit.

Workable gives its radio inputs per-session random ids, so the yes/no answers
are clicked by walking from the question text to the first YES/NO label after
it rather than by id.
"""
import os, sys, json
from playwright.sync_api import sync_playwright
import answers as A

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
ARGS = ["--no-sandbox", "--disable-quic", "--ssl-version-max=tls1.2"]

XET_COVER = (
    "Systems engineer in Vilnius. I wrote a ring-0 operating system in Rust with a "
    "content-addressed store and an NVMe driver in it, and an inference engine that "
    "streams weights off disk so a 2.78-trillion-parameter model runs on 8.24 GB of "
    "RAM. Both are the shape of xet-core's problem from the other side. Everything is "
    f"public: {A.SITE}"
)

FORMS = {
    "hf-xet": {
        "url": "https://apply.workable.com/huggingface/j/F4C096B22E/apply/",
        "text": [
            ("#firstname", A.FIRST), ("#lastname", A.LAST), ("#email", A.EMAIL),
            ("#CA_47143", A.GH), ("#CA_47383", A.LI),
            ("#cover_letter", XET_COVER),
            ("#QA_12194462", A.HF_WHY),
            ("#QA_12194463", A.HF_LOWLEVEL),
            ("#QA_12194464", A.HF_ONTOP),
        ],
        "yesno": [
            ("eligible to work in the country", "YES"),
            ("true and your own", "YES"),
            ("exact phrase", "YES"),
            ("hands-on experience in low-level", "YES"),
        ],
        "hear": ("#CA_10629", "Your careers page"),
    },
    "hf-wildcard": {
        "url": "https://apply.workable.com/huggingface/j/0BD8C06DB3/apply/",
        "text": [
            ("#firstname", A.FIRST), ("#lastname", A.LAST), ("#email", A.EMAIL),
            ("#phone", A.PHONE),
            ("#CA_47143", A.GH), ("#CA_47383", A.LI),
            ("#QA_8723547", A.HF_WILD_WHY),
            ("#QA_5500738", A.HF_WILD_PROJECT),
        ],
        "yesno": [],
        "hear": ("#CA_10629", "Your careers page"),
    },
}


def run(name, submit=False):
    cfg = FORMS[name]
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=ARGS,
                              proxy={"server": os.environ.get("HTTPS_PROXY")})
        pg = b.new_page(viewport={"width": 1280, "height": 1600})
        pg.goto(cfg["url"], wait_until="domcontentloaded", timeout=90000)
        pg.wait_for_timeout(4000)

        for label in ("Decline all", "Accept all"):
            try:
                pg.click(f"button:has-text('{label}')", timeout=4000)
                print(f"  cookies: {label}")
                break
            except Exception:
                pass

        try:
            pg.set_input_files("input[type=file]", A.RESUME)
            pg.wait_for_timeout(4000)
            print("  ok resume")
        except Exception as e:
            print("  FAILED resume:", str(e).splitlines()[0][:100])

        for sel, val in cfg["text"]:
            try:
                pg.fill(sel, val)
                print(f"  ok {sel}")
            except Exception as e:
                print(f"  FAILED {sel}: {str(e).splitlines()[0][:100]}")

        for fragment, answer in cfg["yesno"]:
            try:
                q = pg.locator(f"text={fragment}").first
                q.scroll_into_view_if_needed()
                lab = q.locator(f"xpath=following::label[normalize-space()='{answer}'][1]")
                lab.click(timeout=8000)
                print(f"  ok radio '{fragment[:28]}' -> {answer}")
            except Exception as e:
                print(f"  FAILED radio '{fragment[:28]}': {str(e).splitlines()[0][:90]}")

        try:
            pg.fill(cfg["hear"][0], cfg["hear"][1])
        except Exception:
            pass

        # Notice period is a select; take the option that means "now" if it exists.
        try:
            opts = pg.eval_on_selector_all("select option", "els => els.map(e => e.text)")
            pick = next((o for o in opts if "imm" in o.lower() or "less than" in o.lower()), None)
            if pick:
                pg.select_option("select", label=pick)
                print("  ok notice period:", pick)
        except Exception as e:
            print("  notice period skipped:", str(e).splitlines()[0][:80])

        pg.wait_for_timeout(1200)
        pg.screenshot(path=f"filled-{name}.png", full_page=True)
        state = pg.evaluate("""() => Array.from(document.querySelectorAll('input,textarea'))
            .filter(e => e.type !== 'hidden' && !e.id.startsWith('g-recaptcha'))
            .map(e => ({f: e.id || e.name, t: e.type,
                        v: e.type === 'file' ? (e.files && e.files.length ? e.files[0].name : '(empty)')
                          : (e.type === 'radio' || e.type === 'checkbox') ? e.checked
                          : (e.value || '').slice(0, 50)}))
            .filter(x => x.v !== false)""")
        print(json.dumps(state)[:1600])

        if submit:
            pg.click("button:has-text('Submit application')")
            pg.wait_for_timeout(10000)
            pg.screenshot(path=f"submitted-{name}.png", full_page=True)
            print("AFTER SUBMIT:", pg.inner_text("body")[:400].replace("\n", " | "))
        b.close()


if __name__ == "__main__":
    for n in sys.argv[1:]:
        submit = n.endswith(":submit")
        n = n.split(":")[0]
        print(f"===== {n}{' SUBMITTING' if submit else ''}")
        run(n, submit)
