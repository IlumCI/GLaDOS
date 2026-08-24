"""Load an application form and dump every field it asks for.

Reads the rendered DOM rather than the job board's API, because what matters is
the form the applicant actually faces: which fields are required, which are file
uploads, and which are free-text boxes that need a written answer.
"""
import os, sys, json
from playwright.sync_api import sync_playwright

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

JS = """
() => {
  const out = [];
  const labelFor = (el) => {
    if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
    if (el.id) {
      const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (l) return l.innerText.trim();
    }
    let p = el.closest('div,fieldset,section');
    for (let i = 0; i < 4 && p; i++) {
      const l = p.querySelector('label,legend');
      if (l && l.innerText.trim()) return l.innerText.trim().slice(0, 120);
      p = p.parentElement;
    }
    return '';
  };
  document.querySelectorAll('input,textarea,select,[contenteditable="true"]').forEach(el => {
    const type = el.tagName.toLowerCase() === 'input' ? el.type : el.tagName.toLowerCase();
    if (type === 'hidden') return;
    out.push({
      tag: el.tagName.toLowerCase(),
      type,
      name: el.name || '',
      id: el.id || '',
      required: el.required || el.getAttribute('aria-required') === 'true',
      placeholder: el.placeholder || '',
      label: labelFor(el),
      options: el.tagName.toLowerCase() === 'select'
        ? Array.from(el.options).map(o => o.text).slice(0, 12) : undefined,
    });
  });
  return {fields: out, title: document.title, text: document.body.innerText.slice(0, 1200)};
}
"""


def main(url, shot):
    with sync_playwright() as p:
        # Outbound HTTPS in this container goes through the agent proxy; Chromium
        # has to be told, and it reads the proxy CA from the NSS store the image
        # already set up.
        b = p.chromium.launch(
            executable_path=CHROME,
            # TLS 1.3 ClientHellos are reset by this container's MITM proxy; capping
            # at 1.2 keeps verification on and gets through.
            args=["--no-sandbox", "--disable-quic", "--ssl-version-max=tls1.2"],
            proxy={"server": os.environ.get("HTTPS_PROXY", "http://127.0.0.1:33069")},
        )
        pg = b.new_page(viewport={"width": 1280, "height": 1400})
        pg.goto(url, wait_until="networkidle", timeout=90000)
        pg.wait_for_timeout(2500)
        data = pg.evaluate(JS)
        pg.screenshot(path=shot, full_page=True)
        b.close()
    print(json.dumps(data, indent=1)[:6000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
