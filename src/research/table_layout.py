"""Read explicit HTML column groups without guessing the meaning of empty cells."""
from html.parser import HTMLParser
import re


class _Table(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = self.cell = None
        self.inside = self.finished = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.inside or self.finished:
                raise ValueError("multiple tables")
            self.inside = True
        elif self.inside and tag == "tr":
            if self.row is not None or len(self.rows) >= 132:
                raise ValueError("invalid row")
            self.row = []
        elif self.inside and tag in {"td", "th"}:
            if self.row is None or self.cell is not None or len(self.row) >= 32:
                raise ValueError("invalid cell")
            attrs = dict(attrs)
            sizes = [attrs.get(key, "1") for key in ("rowspan", "colspan")]
            if not all(v and re.fullmatch(r"[1-9][0-9]?", v) and int(v) <= 32 for v in sizes):
                raise ValueError("invalid span")
            self.cell = [tag, *map(int, sizes), []]
        elif self.cell and tag == "br":
            self.cell[-1].append(" ")

    def handle_data(self, data):
        if self.cell:
            self.cell[-1].append(data)

    def handle_endtag(self, tag):
        if not self.inside:
            return
        if tag in {"th", "td"}:
            if not self.cell or self.cell[0] != tag:
                raise ValueError("unpaired cell")
            kind, down, across, parts = self.cell
            self.row.append((kind, down, across, " ".join("".join(parts).split())))
            self.cell = None
        elif tag == "tr":
            if not self.row or self.cell:
                raise ValueError("incomplete row")
            self.rows.append(self.row)
            self.row = None
        elif tag == "table":
            if self.row is not None:
                raise ValueError("incomplete table")
            self.inside, self.finished = False, True


def html_table(text):
    parser = _Table()
    try:
        parser.feed(text)
        parser.close()
    except ValueError:
        return None
    if not parser.finished:
        return None
    grid, body = {}, []
    depth = 0
    for row in parser.rows:
        if not body and all(kind == "th" for kind, _, _, _ in row):
            col = 0
            for _, down, across, label in row:
                while (depth, col) in grid:
                    col += 1
                if col + across > 32 or depth + down > 8:
                    return None
                for r in range(depth, depth+down):
                    for c in range(col, col+across):
                        if (r, c) in grid:
                            return None
                        grid[r, c] = (depth, col, label)
                col += across
            depth += 1
        else:
            body.append(row)
    if not depth or not body:
        return None
    width = max(c for _, c in grid) + 1
    if set(grid) != {(r,c) for r in range(depth) for c in range(width)}:
        return None
    slots, extents = {}, {}
    for r, row in enumerate(body):
        col = 0
        for kind, down, across, value in row:
            while (r, col) in slots:
                col += 1
            if (col + across > width or r + down > len(body) or (col and kind != "td")
                    or (not col and across != 1)):
                return None
            for rr in range(r, r+down):
                for cc in range(col, col+across):
                    if (rr, cc) in slots:
                        return None
                    slots[rr, cc] = value if cc == 0 or (rr, cc) == (r, col) else ""
            if col and (across > 1 or down > 1):
                extents[r, col] = (down, across)
            col += across
    if len(slots) != len(body)*width:
        return None
    values = [[slots[r,c] for c in range(width)] for r in range(len(body))]
    headers = [" / ".join(v for _, _, v in dict.fromkeys(grid[r,c] for r in range(depth)) if v)
               for c in range(width)]
    return headers, values, extents
