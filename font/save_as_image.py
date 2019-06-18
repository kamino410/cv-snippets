from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.svgPathPen import SVGPathPen
from textwrap import dedent

from cairosvg import svg2png
# export LC_ALL=en_US.UTF-8
# export LANG=en_US.UTF-8


def save_as_svg(font, char, output_path):
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()

    # グリフのアウトラインを SVGPathPen でなぞる
    glyph = get_glyph(glyph_set, cmap, char)
    svg_path_pen = SVGPathPen(glyph_set)
    glyph.draw(svg_path_pen)

    # メトリクスを取得
    ascender = font['OS/2'].sTypoAscender
    descender = font['OS/2'].sTypoDescender
    width = glyph.width
    height = ascender - descender

    content = dedent(f'''\
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {-ascender} {width} {height}">
            <g transform="scale(1, -1)">
                <path d="{svg_path_pen.getCommands()}"/>
            </g>
        </svg>
    ''')

    with open(output_path, 'w') as f:
        f.write(content)


def get_glyph(glyph_set, cmap, char):
    glyph_name = cmap[ord(char)]
    return glyph_set[glyph_name]


font = TTFont('/System/Library/Fonts/Helvetica.ttc', fontNumber=1)

save_as_svg(font, 'A', 'A.svg')

svg2png(url='A.svg', write_to='A.png', parent_width=140, parent_height=200)
