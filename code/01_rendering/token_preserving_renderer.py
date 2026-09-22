#!/usr/bin/env python3
"""Deterministic code-image renderer with token-preserving visual soft wrap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.styles import get_style_by_name
from pygments.util import ClassNotFound


@dataclass(frozen=True)
class Cell:
    char: str
    color: str


@dataclass(frozen=True)
class VisualRow:
    source_line: int
    show_line_number: bool
    cells: tuple[Cell, ...]


def normalize_code(code: str) -> str:
    return str(code).replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def _hex(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    return value if value.startswith("#") else f"#{value}"


def _lexer(language: str):
    try:
        return get_lexer_by_name(language)
    except ClassNotFound:
        return TextLexer()


def tokenized_lines(code: str, language: str, style_name: str, tab_width: int = 4) -> list[list[Cell]]:
    """Lex original source once, then return styled display cells per logical line."""
    style = get_style_by_name(style_name)
    lines: list[list[Cell]] = [[]]
    column = 0
    for token_type, value in lex(normalize_code(code), _lexer(language)):
        color = _hex(style.style_for_token(token_type).get("color"), "#000000")
        for char in value:
            if char == "\n":
                lines.append([])
                column = 0
            elif char == "\t":
                spaces = tab_width - (column % tab_width)
                lines[-1].extend(Cell(" ", color) for _ in range(spaces))
                column += spaces
            else:
                lines[-1].append(Cell(char, color))
                column += 1
    # Pygments commonly emits one terminal newline; it is not a source line.
    if len(lines) > 1 and not lines[-1] and not normalize_code(code).endswith("\n"):
        lines.pop()
    return lines


def layout_rows(
    code: str,
    language: str,
    style_name: str,
    wrap_column: int,
    tab_width: int = 4,
    remove_indent: bool = False,
    remove_blank_lines: bool = False,
) -> list[VisualRow]:
    logical = tokenized_lines(code, language, style_name, tab_width)
    rows: list[VisualRow] = []
    for line_number, original_cells in enumerate(logical, 1):
        cells = list(original_cells)
        if remove_indent:
            while cells and cells[0].char == " ":
                cells.pop(0)
        if remove_blank_lines and not any(cell.char.strip() for cell in cells):
            continue
        chunks = [cells[index:index + wrap_column] for index in range(0, len(cells), wrap_column)] or [[]]
        for chunk_index, chunk in enumerate(chunks):
            rows.append(VisualRow(line_number, chunk_index == 0, tuple(chunk)))
    return rows or [VisualRow(1, True, tuple())]


def render_code(
    code: str,
    language: str,
    output: Path,
    font_path: Path,
    style_name: str,
    font_size: int,
    wrap_column: int,
    line_numbers: bool = True,
    tab_width: int = 4,
    image_pad: int = 10,
    line_gap: int = 6,
    remove_indent: bool = False,
    remove_blank_lines: bool = False,
) -> dict[str, int | str | bool]:
    style = get_style_by_name(style_name)
    background = _hex(getattr(style, "background_color", None), "#ffffff")
    foreground = "#eeeeee" if sum(ImageColor(background)) < 300 else "#666666"
    font = ImageFont.truetype(str(font_path), font_size)
    bbox = font.getbbox("M")
    char_width = max(1, int(round(font.getlength("M"))))
    glyph_height = bbox[3] - bbox[1]
    row_height = glyph_height + line_gap
    rows = layout_rows(
        code, language, style_name, wrap_column, tab_width,
        remove_indent=remove_indent, remove_blank_lines=remove_blank_lines,
    )
    source_lines = len(tokenized_lines(code, language, style_name, tab_width))
    digits = max(2, len(str(source_lines)))
    gutter_width = (digits + 2) * char_width if line_numbers else 0
    width = image_pad * 2 + gutter_width + wrap_column * char_width
    height = image_pad * 2 + len(rows) * row_height
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    if line_numbers:
        gutter_fill = blend(background, "#808080", 0.10)
        draw.rectangle((0, 0, image_pad + gutter_width - char_width // 2, height), fill=gutter_fill)
    for row_index, row in enumerate(rows):
        y = image_pad + row_index * row_height - bbox[1]
        if line_numbers and row.show_line_number:
            number = str(row.source_line).rjust(digits)
            draw.text((image_pad, y), number, font=font, fill=foreground)
        x = image_pad + gutter_width
        for cell in row.cells:
            if cell.char != " ":
                draw.text((x, y), cell.char, font=font, fill=cell.color)
            x += char_width
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return {
        "image_width": width,
        "image_height": height,
        "source_lines": source_lines,
        "visual_rows": len(rows),
        "continuation_rows": sum(not row.show_line_number for row in rows),
        "char_width": char_width,
        "glyph_height": glyph_height,
        "row_height": row_height,
        "background": background,
        "remove_indent": remove_indent,
        "remove_blank_lines": remove_blank_lines,
    }


def ImageColor(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def blend(left: str, right: str, right_weight: float) -> str:
    a, b = ImageColor(left), ImageColor(right)
    mixed = tuple(round(x * (1 - right_weight) + y * right_weight) for x, y in zip(a, b))
    return "#" + "".join(f"{value:02x}" for value in mixed)
