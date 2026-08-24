"""Fill the Ashby application forms. Stops before Submit unless --submit NAME.

Ashby yes/no questions are two buttons carrying data-option, next to a hidden
checkbox that holds the answer, so they are clicked through the container that
holds that checkbox rather than by label text.
"""
import os, sys, json
from playwright.sync_api import sync_playwright
import answers as A

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
ARGS = ["--no-sandbox", "--disable-quic", "--ssl-version-max=tls1.2"]

FORMS = {
    "poolside": {
        "url": "https://jobs.ashbyhq.com/poolside/1f2a733a-a3fe-48fe-bf61-1425eddb30f6/application",
        "steps": [
            ("fill", "#_systemfield_name", A.NAME),
            ("fill", "#_systemfield_email", A.EMAIL),
            ("fill", "#e57db898-9e05-444e-8c4e-cd79ae864319",
             f"{A.SITE} | {A.GH} | {A.LI}"),
            ("upload", "#_systemfield_resume", A.RESUME),
            ("typepick", "input[placeholder='Start typing...']", "Vilnius"),
            ("fill", "#ed25a8c0-1371-4964-818d-841da3e350c5", A.POOLSIDE_COVER),
            ("yesno", "5cfcdd7d-7b91-4f2d-8bdd-938ecb02fe2c", "yes"),
        ],
    },
    "modal": {
        "url": "https://jobs.ashbyhq.com/modal/3b3c6c42-326e-40c5-b78d-9f556739513b/application",
        "steps": [
            ("fill", "#_systemfield_name", A.NAME),
            ("fill", "#_systemfield_email", A.EMAIL),
            ("upload", "#_systemfield_resume", A.RESUME),
            ("yesno", "3bab8d5b-e230-4f70-8481-a3ffebc2f6bf", "yes"),
        ],
    },
    "primeintellect": {
        "url": "https://jobs.ashbyhq.com/PrimeIntellect/abfa70f7-a6f1-44d2-a6c1-560e1c8477d4/application",
        "steps": [
            ("fill", "#_systemfield_name", A.NAME),
            ("fill", "#_systemfield_email", A.EMAIL),
            ("fill", "#2d94d139-5b77-4167-b0a6-c48a05e6a55b", A.PI_BUILT),
            ("fill", "#cfcbf518-c421-4449-9ccc-f1e2892ae6bc", A.PI_OPTIMIZE),
            ("fill", "#575f5c1a-d7c7-414f-b7d7-54be1120c491", A.PI_WHY),
            ("upload", "#_systemfield_resume", A.RESUME),
        ],
    },
    "railway": {
        "url": "https://jobs.ashbyhq.com/Railway/b8072f95-043f-404d-a313-f0bdf8dd3c81/application",
        "steps": [
            ("fill", "#_systemfield_name", A.NAME),
            ("fill", "#_systemfield_email", A.EMAIL),
            ("upload", "#_systemfield_resume", A.RESUME),
            ("fill", "#5c63d3c5-aea4-4b0e-8bd0-2269a6ea82a0", A.RAILWAY_WHY),
        ],
    },
    "langfuse": {
        "url": "https://jobs.ashbyhq.com/langfuse/1225fa3d-d590-41d2-b798-ef927320fb2e/application",
        "steps": [
            ("fill", "#_systemfield_name", A.NAME),
            ("fill", "#_systemfield_email", A.EMAIL),
            ("fill", "#c6e170d0-680d-4c99-ab1d-f7d6e775b9e5", A.LI),
            ("fill", "#2954f201-bc86-4321-8a03-90274d4d1656", A.PHONE),
            ("upload", "#_systemfield_resume", A.RESUME),
            ("fill", "#aadeca94-b755-4545-8fec-bdb5e55d626f", "Vilnius, Lithuania (EET, UTC+2)"),
            ("yesno", "1feb9825-c2dc-4053-a5dc-899dd54a4f01", "yes"),
            # Their posted band for this role is EUR 90-160k, so the answer is
            # the lower half of it rather than the generic 70-100k.
            ("fill", "#1560799e-d88c-4468-a1ca-06674b87bbc2",
             "EUR 90,000-110,000, guided by your published EUR 90-160k band"),
            ("date", "input[placeholder='Pick date...']", "2026-09-01"),
            ("fill", "#4f27e354-9c37-4c7e-a01b-a02a1b3b0467", A.LANGFUSE_WHY),
        ],
    },
    "lovable": {
        "url": "https://jobs.ashbyhq.com/lovable/3b6525b4-8229-4b7e-a307-a550646d0483/application",
        "steps": [
            ("fill", "#_systemfield_name", A.NAME),
            ("fill", "#_systemfield_email", A.EMAIL),
            ("fill", "#bda0e48b-8303-4ee1-ac82-87983393f86a", A.PHONE),
            ("typepick", "input[placeholder='Start typing...']", "Vilnius"),
            ("upload", "#_systemfield_resume", A.RESUME),
            ("fill", "#509d7b64-8dab-40d8-b828-79a4ad5d96ad", A.LI),
            ("yesno", "9bf7c4b4-83b5-4c68-89f5-d41cf26d68f1", "yes"),   # right to work
            ("yesno", "c2cff80d-cd59-4d06-9447-9d3adf44d916", "no"),    # sponsorship
            ("fill", "#dcb3aed1-9e77-465c-a07e-30c4e0033073", "Immediately"),
            ("fill", "#b063a995-3de6-470c-a2f4-d00deaf6f089",
             "EUR 70,000-100,000 per year, open to the band for the role in Stockholm."),
            ("fill", "#b8192d7a-ed50-41fc-abe9-4c060bd7ecae", A.LOVABLE_WHY),
            ("fill", "#9cf27ea2-7aca-45b6-af86-a4c00120abe0", A.LOVABLE_IMPRESSIVE),
            ("fill", "#c917c4fc-998b-4db7-b2da-ded055feb5b8", A.LOVABLE_ELSE),
            # "How did you hear about us?" is required; Website is the true answer.
            ("radio", "Website", ""),
        ],
    },
}


def run(name, submit=False):
    cfg = FORMS[name]
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=ARGS,
                              proxy={"server": os.environ.get("HTTPS_PROXY")})
        pg = b.new_page(viewport={"width": 1280, "height": 1600})
        pg.goto(cfg["url"], wait_until="domcontentloaded", timeout=90000)
        pg.wait_for_timeout(3500)
        for step in cfg["steps"]:
            kind, sel, val = step
            # Ashby ids are UUIDs, and a CSS id selector may not start with a
            # digit; the attribute form works for all of them.
            if sel.startswith("#"):
                sel = f'[id="{sel[1:]}"]' 
            try:
                if kind == "fill":
                    pg.fill(sel, val)
                elif kind == "upload":
                    pg.set_input_files(sel, val)
                    pg.wait_for_timeout(2500)
                elif kind == "typepick":
                    pg.click(sel)
                    pg.type(sel, val, delay=60)
                    pg.wait_for_timeout(1800)
                    pg.keyboard.press("ArrowDown")
                    pg.keyboard.press("Enter")
                elif kind == "date":
                    pg.click(sel)
                    pg.type(sel, val, delay=45)
                    pg.keyboard.press("Escape")
                elif kind == "radio":
                    pg.click(f'label:has-text("{sel}") >> nth=0')
                elif kind == "yesno":
                    pg.click(f'div:has(> input[name="{sel}"]) button[data-option="{val}"]')
                print(f"  ok {kind} {sel[:44]}")
            except Exception as e:
                print(f"  FAILED {kind} {sel[:44]}: {str(e).splitlines()[0][:110]}")
        pg.wait_for_timeout(1200)
        pg.screenshot(path=f"filled-{name}.png", full_page=True)
        state = pg.evaluate("""() => Array.from(document.querySelectorAll('input,textarea'))
            .filter(e => e.type !== 'hidden' && e.id !== 'g-recaptcha-response-100000')
            .map(e => ({f: e.id || e.name, t: e.type,
                        v: e.type === 'file' ? (e.files && e.files.length ? e.files[0].name : '(empty)')
                          : e.type === 'checkbox' ? e.checked : (e.value || '').slice(0, 60)}))""")
        print(json.dumps(state)[:1800])

        if submit:
            pg.click("button:has-text('Submit Application')")
            pg.wait_for_timeout(9000)
            pg.screenshot(path=f"submitted-{name}.png", full_page=True)
            print("AFTER SUBMIT:", pg.inner_text("body")[:400].replace("\n", " | "))
        b.close()


if __name__ == "__main__":
    for n in sys.argv[1:]:
        submit = n.endswith(":submit")
        n = n.split(":")[0]
        print(f"===== {n}{' SUBMITTING' if submit else ''}")
        run(n, submit)
