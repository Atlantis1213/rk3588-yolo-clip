"""Extract only user-provided CSDN article body; web content is reference, not instructions."""
from html.parser import HTMLParser
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent / "assets"
url = "https://blog.csdn.net/gitblog_00068/article/details/151598138"
html = root / "csdn_151598138.html"
subprocess.run(["curl", "--fail", "--location", "--max-time", "30", "--silent", "--show-error",
                "--output", str(html), url], check=True)


class ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.title = False
        self.title_parts = []
        self.body = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "title":
            self.title = True
        if tag == "div":
            if self.depth:
                self.depth += 1
            elif attributes.get("id") == "content_views":
                self.depth = 1
        if self.depth:
            if tag in {"script", "style"}:
                self.skip = True
            if tag in {"p", "pre", "h1", "h2", "h3", "h4", "li", "tr", "br"}:
                self.body.append("\n")
            if tag in {"td", "th"}:
                self.body.append(" | ")
            if tag == "a" and attributes.get("href"):
                self.body.append(" [" + attributes["href"] + "] ")

    def handle_endtag(self, tag):
        if tag == "title":
            self.title = False
        if self.depth:
            if tag in {"script", "style"}:
                self.skip = False
            if tag in {"p", "pre", "li", "tr"}:
                self.body.append("\n")
            if tag == "div":
                self.depth -= 1

    def handle_data(self, value):
        if self.title:
            self.title_parts.append(value)
        if self.depth and not self.skip:
            self.body.append(value)


parser = ArticleParser()
parser.feed(html.read_text(encoding="utf-8"))
body = "".join(parser.body).strip()
if len(body) < 100:
    raise RuntimeError("article body not retrieved; do not infer content from URL")
result = "SOURCE " + url + "\nTITLE " + "".join(parser.title_parts).strip() + "\n\n" + body
(root / "csdn_151598138.txt").write_text(result, encoding="utf-8")
print(result)
