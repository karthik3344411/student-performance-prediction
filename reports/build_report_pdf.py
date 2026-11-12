"""
Build reports/Project_Report.pdf from reports/project_report.md

Steps: Markdown -> styled HTML (reports/project_report.html) -> PDF printed by
Microsoft Edge in headless mode (installed on every Windows 10/11 PC).

Run (after editing your details on the title page of project_report.md):
    python reports/build_report_pdf.py
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

REPORTS = Path(__file__).resolve().parent
MD_FILE = REPORTS / "project_report.md"
HTML_FILE = REPORTS / "project_report.html"
PDF_FILE = REPORTS / "Project_Report.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm;
        @bottom-center { content: "Page " counter(page); font: 9pt 'Segoe UI', Arial, sans-serif; color: #898781; }
        @bottom-left { content: "CSE570 - Student Performance Prediction"; font: 9pt 'Segoe UI', Arial, sans-serif; color: #898781; } }
@page :first { @bottom-center { content: none; } @bottom-left { content: none; } }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1a1a19; }
h1, h2, h3 { color: #184f95; line-height: 1.25; }
h2 { font-size: 17pt; border-bottom: 2px solid #2a78d6; padding-bottom: 4px; margin-top: 0; break-before: page; }
h3 { font-size: 12.5pt; margin: 16px 0 6px; break-after: avoid; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9.2pt; break-inside: avoid; }
th, td { border: 1px solid #c3c2b7; padding: 4px 7px; vertical-align: top; text-align: left; }
th { background: #e8f1fc; color: #0d366b; }
tr:nth-child(even) td { background: #fafaf8; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f0efec; padding: 0 3px; border-radius: 3px; }
img { max-width: 100%; }
.fig { text-align: center; break-inside: avoid; margin: 10px 0 14px; }
.fig img { max-height: 118mm; }
.fig p { text-align: center; margin: 2px 0; }
em { color: #52514e; }
p:has(+ table) { break-after: avoid-page; }   /* keep "Table N." captions with their table */
ul, ol { break-inside: avoid; }
/* two figures side by side (markdown tables whose cells contain images) */
table.figs, table.figs td, table.figs th { border: none; background: none !important; }
table.figs td, table.figs th { width: 50%; text-align: center; font-weight: normal; padding: 2px 4px; }
table.figs img { max-height: 80mm; }
/* title page */
.title-page { height: 255mm; display: flex; flex-direction: column; justify-content: center; text-align: center; }
.title-page .uni { font-size: 20pt; font-weight: 700; color: #eb6834; letter-spacing: 1px; margin: 0; text-align: center; }
.title-page .course { font-size: 12pt; color: #52514e; margin: 6px 0 40px; text-align: center; }
.title-page .main-title { font-size: 26pt; color: #184f95; margin: 0 0 8px; }
.title-page .subtitle { font-size: 13pt; color: #52514e; margin: 0 0 40px; text-align: center; }
.title-page table { width: 85%; margin: 0 auto; font-size: 11pt; }
.title-page table thead { display: none; }
.title-page td:first-child { width: 30%; white-space: nowrap; }
.title-page td:last-child { word-break: break-all; }
"""


def build_html():
    md_text = MD_FILE.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "md_in_html", "attr_list", "toc"])
    # Tables whose header row contains images are "two figures side by side" layouts
    body = re.sub(r"<table>(\s*<thead>.*?</thead>)",
                  lambda m: ('<table class="figs">' if "<img" in m.group(1) else "<table>") + m.group(1),
                  body, flags=re.S)
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Student Performance Prediction - CSE570 Report</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    HTML_FILE.write_text(html, encoding="utf-8")
    return HTML_FILE


def find_browser():
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return shutil.which("msedge") or shutil.which("chrome") or shutil.which("chromium")


def build_pdf(html_path):
    browser = find_browser()
    if browser is None:
        sys.exit("Microsoft Edge or Google Chrome is needed to print the PDF. "
                 f"You can also open {html_path} in a browser and use Print -> Save as PDF.")
    profile = REPORTS / ".pdf_browser_profile"
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--user-data-dir={profile}", f"--print-to-pdf={PDF_FILE}",
                    html_path.as_uri()], check=True, timeout=180)
    shutil.rmtree(profile, ignore_errors=True)
    return PDF_FILE


if __name__ == "__main__":
    html = build_html()
    pdf = build_pdf(html)
    print(f"HTML: {html}\nPDF : {pdf} ({pdf.stat().st_size / 1e6:.1f} MB)")
