import concurrent.futures
import builtins
import logging
import re
import unicodedata
from enum import Enum
from string import Template
from typing import Dict

import numpy as np
from pdfminer.converter import PDFConverter
from pdfminer.layout import LTChar, LTFigure, LTLine, LTPage
from pdfminer.pdffont import PDFCIDFont, PDFUnicodeNotDefined
from pdfminer.pdfinterp import PDFGraphicState, PDFResourceManager
from pdfminer.utils import apply_matrix_pt, mult_matrix
from pymupdf import Font
from tenacity import retry, stop_after_attempt, wait_fixed

from pdf2zh.translator import (
    AnythingLLMTranslator,
    ArgosTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DeepseekTranslator,
    DifyTranslator,
    GeminiTranslator,
    GoogleTranslator,
    GrokTranslator,
    GroqTranslator,
    MiniMaxTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAIlikedTranslator,
    OpenAITranslator,
    QwenMtTranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
    X302AITranslator,
)

log = logging.getLogger(__name__)


class PDFConverterEx(PDFConverter):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
    ) -> None:
        PDFConverter.__init__(self, rsrcmgr, None, "utf-8", 1, None)

    def begin_page(self, page, ctm) -> None:
        # 重载替换 cropbox
        x0, y0, x1, y1 = page.cropbox
        x0, y0 = apply_matrix_pt(ctm, (x0, y0))
        x1, y1 = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.cur_item = LTPage(page.pageno, mediabox)

    def end_page(self, page):
        # 重载返回指令流
        return self.receive_layout(self.cur_item)

    def begin_figure(self, name, bbox, matrix) -> None:
        # 重载设置 pageid
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))
        self.cur_item.pageid = self._stack[-1].pageid

    def end_figure(self, _: str) -> None:
        # 重载返回指令流
        fig = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)
        return self.receive_layout(fig)

    def render_char(
        self,
        matrix,
        font,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs,
        graphicstate: PDFGraphicState,
    ) -> float:
        # 重载设置 cid 和 font
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        item = LTChar(
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        )
        self.cur_item.add(item)
        item.cid = cid  # hack 插入原字符编码
        item.font = font  # hack 插入原字符字体
        return item.adv


class Paragraph:
    def __init__(
        self,
        y,
        x,
        x0,
        x1,
        y0,
        y1,
        size,
        brk,
        font_style="regular",
        color=(0.0,),
    ):
        self.y: float = y  # 初始纵坐标
        self.x: float = x  # 初始横坐标
        self.x0: float = x0  # 左边界
        self.x1: float = x1  # 右边界
        self.y0: float = y0  # 上边界
        self.y1: float = y1  # 下边界
        self.size: float = size  # 字体大小
        self.brk: bool = brk  # 换行标记
        self.font_style: str = font_style
        self.color = color
        # Character offsets where the source PDF explicitly started a new
        # line. Translation must not collapse these author-supplied breaks.
        self.break_positions: list[int] = []


TOC_ENTRY_RE = re.compile(
    r"^(?P<label>.*?)(?P<leaders>\.{4,})(?P<page>\s*\d+\s*)$",
    re.DOTALL,
)
CODE_LINE_RE = re.compile(r"^N\d+\b", re.IGNORECASE)
TECHNICAL_BOLD_TOKEN_RE = re.compile(
    r"(?:\d+(?:[.,]\d+)*|\.[A-Za-z0-9]+|[GMNT]\d+(?:[.,]\d+)*)",
    re.IGNORECASE,
)
BULLET_GLYPHS = frozenset("•◦▪▫‣⁃➢➤➔→►▸")


def is_fixed_layout_line(text: str) -> bool:
    """Return True for rows whose coordinates carry semantic meaning."""
    stripped = text.strip()
    return bool(
        TOC_ENTRY_RE.match(stripped)
        or CODE_LINE_RE.match(stripped)
        or (stripped.startswith("{") and stripped.endswith("}"))
    )


# fmt: off
class TranslateConverter(PDFConverterEx):
    def __init__(
        self,
        rsrcmgr,
        vfont: str = None,
        vchar: str = None,
        thread: int = 0,
        layout={},
        lang_in: str = "",
        lang_out: str = "",
        service: str = "",
        noto_name: str = "",
        noto: Font = None,
        envs: Dict = None,
        prompt: Template = None,
        ignore_cache: bool = False,
    ) -> None:
        super().__init__(rsrcmgr)
        self.vfont = vfont
        self.vchar = vchar
        self.thread = thread
        self.layout = layout
        self.noto_name = noto_name
        self.noto = noto
        self.translator: BaseTranslator = None
        # e.g. "ollama:gemma2:9b" -> ["ollama", "gemma2:9b"]
        param = service.split(":", 1)
        service_name = param[0]
        service_model = param[1] if len(param) > 1 else None
        if not envs:
            envs = {}
        for translator in [GoogleTranslator, BingTranslator, DeepLTranslator, DeepLXTranslator, OllamaTranslator, XinferenceTranslator, AzureOpenAITranslator,
                           OpenAITranslator, ZhipuTranslator, ModelScopeTranslator, SiliconTranslator, GeminiTranslator, AzureTranslator, TencentTranslator, DifyTranslator, AnythingLLMTranslator, ArgosTranslator, GrokTranslator, GroqTranslator, DeepseekTranslator, MiniMaxTranslator, OpenAIlikedTranslator, QwenMtTranslator, X302AITranslator]:
            if service_name == translator.name:
                self.translator = translator(lang_in, lang_out, service_model, envs=envs, prompt=prompt, ignore_cache=ignore_cache)
        if not self.translator:
            raise ValueError("Unsupported translation service")

    def receive_layout(self, ltpage: LTPage):
        # 段落
        sstk: list[str] = []            # 段落文字栈
        pstk: list[Paragraph] = []      # 段落属性栈
        vbkt: int = 0                   # 段落公式括号计数
        # 公式组
        vstk: list[LTChar] = []         # 公式符号组
        vlstk: list[LTLine] = []        # 公式线条组
        vfix: float = 0                 # 公式纵向偏移
        # 公式组栈
        var: list[list[LTChar]] = []    # 公式符号组栈
        varl: list[list[LTLine]] = []   # 公式线条组栈
        varf: list[float] = []          # 公式纵向偏移栈
        vlen: list[float] = []          # 公式宽度栈
        # 全局
        lstk: list[LTLine] = []         # 全局线条栈
        xt: LTChar = None               # 上一个字符
        xt_cls: int = -1                # 上一个字符所属段落，保证无论第一个字符属于哪个类别都可以触发新段落
        vmax: float = ltpage.width / 4  # 行内公式最大宽度
        ops: str = ""                   # 渲染结果

        def font_style_for(font_name: str) -> str:
            source_font = str(font_name).lower()
            is_bold = "bold" in source_font
            is_italic = "italic" in source_font or "oblique" in source_font
            if is_bold and is_italic:
                return "bold_italic"
            if is_bold:
                return "bold"
            if is_italic:
                return "italic"
            return "regular"

        def source_color(char: LTChar):
            color = getattr(getattr(char, "graphicstate", None), "ncolor", None)
            if color is None:
                return (0.0,)
            if isinstance(color, (int, float)):
                return (float(color),)
            try:
                return tuple(float(component) for component in color)
            except (TypeError, ValueError):
                return (0.0,)

        # Preformatted machine listings encode meaning in exact x/y positions.
        # Mark their original glyphs for positional preservation instead of
        # reconstructing them with a wider fallback font.
        fixed_layout_char_ids: set[int] = set()
        technical_bold_tokens: dict[str, str] = {}
        current_line: list[LTChar] = []

        def finish_source_line() -> None:
            if not current_line:
                return
            line_text = "".join(char.get_text() for char in current_line).strip()
            if CODE_LINE_RE.match(line_text) or (
                line_text.startswith("{") and line_text.endswith("}")
            ):
                fixed_layout_char_ids.update(builtins.id(char) for char in current_line)
                return

            # Record inline emphasis separately from coordinate-locked text.
            # Values such as 35, .din and G99 remain part of the sentence sent
            # to the translator, then regain their original emphasis when the
            # translated paragraph is rendered.
            token_chars: list[LTChar] = []

            def finish_token() -> None:
                if not token_chars:
                    return
                token = "".join(char.get_text() for char in token_chars)
                styles = {font_style_for(char.fontname) for char in token_chars}
                if TECHNICAL_BOLD_TOKEN_RE.fullmatch(token) and any(
                    "bold" in style for style in styles
                ):
                    style = next(
                        style for style in styles if "bold" in style
                    )
                    technical_bold_tokens[token] = style

            for line_char in current_line:
                if line_char.get_text().isspace():
                    finish_token()
                    token_chars = []
                else:
                    token_chars.append(line_char)
            finish_token()

        previous_line_char: LTChar | None = None
        for source_child in ltpage:
            if not isinstance(source_child, LTChar):
                continue
            if previous_line_char is not None and source_child.x1 < previous_line_char.x0:
                finish_source_line()
                current_line = []
            current_line.append(source_child)
            previous_line_char = source_child
        finish_source_line()

        def vflag(font: str, char: str):    # 匹配公式（和角标）字体
            if isinstance(font, bytes):     # 不一定能 decode，直接转 str
                try:
                    font = font.decode('utf-8')  # 尝试使用 UTF-8 解码
                except UnicodeDecodeError:
                    font = ""
            font = font.split("+")[-1]      # 字体名截断
            if re.match(r"\(cid:", char):
                return True
            # 基于字体名规则的判定
            if self.vfont:
                if re.match(self.vfont, font):
                    return True
            else:
                if re.match(                                            # latex 字体
                    r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Sym|.*Math|.*Wingdings|.*Webdings|.*Dingbats)",
                    font,
                ):
                    return True
            # 基于字符集规则的判定
            if self.vchar:
                if re.match(self.vchar, char):
                    return True
            else:
                if (
                    char
                    and char != " "                                     # 非空格
                    and (
                        unicodedata.category(char[0])
                        in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"]   # 文字修饰符、数学符号、分隔符号
                        or ord(char[0]) in range(0x370, 0x400)          # 希腊字母
                    )
                ):
                    return True
            return False

        ############################################################
        # A. 原文档解析
        for child in ltpage:
            if isinstance(child, LTChar):
                cur_v = False
                layout = self.layout[ltpage.pageid]
                # ltpage.height 可能是 fig 里面的高度，这里统一用 layout.shape
                h, w = layout.shape
                # 读取当前字符在 layout 中的类别
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                if builtins.id(child) in fixed_layout_char_ids:
                    cls = -1000
                # 锚定文档中 bullet 的位置
                if child.get_text() in BULLET_GLYPHS:
                    cls = 0
                # 判定当前字符是否属于公式
                if (                                                                                        # 判定当前字符是否属于公式
                    cls == 0                                                                                # 1. 类别为保留区域
                    or builtins.id(child) in fixed_layout_char_ids                                          # 1b. preformatted source row
                    or (cls == xt_cls and len(sstk[-1].strip()) > 1 and child.size < pstk[-1].size * 0.79)  # 2. 角标字体，有 0.76 的角标和 0.799 的大写，这里用 0.79 取中，同时考虑首字母放大的情况
                    or vflag(child.fontname, child.get_text())                                              # 3. 公式字体
                    or (child.matrix[0] == 0 and child.matrix[3] == 0)                                      # 4. 垂直字体
                ):
                    cur_v = True
                # 判定括号组是否属于公式
                if not cur_v:
                    if vstk and child.get_text() == "(":
                        cur_v = True
                        vbkt += 1
                    if vbkt and child.get_text() == ")":
                        cur_v = True
                        vbkt -= 1
                if (                                                        # 判定当前公式是否结束
                    not cur_v                                               # 1. 当前字符不属于公式
                    or cls != xt_cls                                        # 2. 当前字符与前一个字符不属于同一段落
                    # or (abs(child.x0 - xt.x0) > vmax and cls != 0)        # 3. 段落内换行，可能是一长串斜体的段落，也可能是段内分式换行，这里设个阈值进行区分
                    # 禁止纯公式（代码）段落换行，直到文字开始再重开文字段落，保证只存在两种情况
                    # A. 纯公式（代码）段落（锚定绝对位置）sstk[-1]=="" -> sstk[-1]=="{v*}"
                    # B. 文字开头段落（排版相对位置）sstk[-1]!=""
                    or (sstk[-1] != "" and abs(child.x0 - xt.x0) > vmax)    # 因为 cls==xt_cls==0 一定有 sstk[-1]==""，所以这里不需要再判定 cls!=0
                ):
                    if vstk:
                        if (                                                # 根据公式右侧的文字修正公式的纵向偏移
                            not cur_v                                       # 1. 当前字符不属于公式
                            and cls == xt_cls                               # 2. 当前字符与前一个字符属于同一段落
                            and child.x0 > max([vch.x0 for vch in vstk])    # 3. 当前字符在公式右侧
                        ):
                            vfix = vstk[0].y0 - child.y0
                        if sstk[-1] == "":
                            xt_cls = -1 # 禁止纯公式段落（sstk[-1]=="{v*}"）的后续连接，但是要考虑新字符和后续字符的连接，所以这里修改的是上个字符的类别
                        sstk[-1] += f"{{v{len(var)}}}"
                        var.append(vstk)
                        varl.append(vlstk)
                        varf.append(vfix)
                        vstk = []
                        vlstk = []
                        vfix = 0
                # 当前字符不属于公式或当前字符是公式的第一个字符
                if not vstk:
                    if cls == xt_cls:               # 当前字符与前一个字符属于同一段落
                        if child.x0 > xt.x1 + 1:    # 添加行内空格
                            sstk[-1] += " "
                        elif child.x1 < xt.x0:      # 原文换行
                            # A table-of-contents row is an independently
                            # positioned record. Merging rows into a paragraph
                            # lets translation move page numbers and leaders
                            # into later lines, destroying the original layout.
                            if is_fixed_layout_line(sstk[-1]):
                                sstk.append("")
                                pstk.append(
                                    Paragraph(
                                        child.y0,
                                        child.x0,
                                        child.x0,
                                        child.x0,
                                        child.y0,
                                        child.y1,
                                        child.size,
                                        False,
                                        font_style_for(child.fontname),
                                        source_color(child),
                                    )
                                )
                            else:
                                pstk[-1].break_positions.append(len(sstk[-1]))
                                sstk[-1] += " "
                                pstk[-1].brk = True
                    else:                           # 根据当前字符构建一个新的段落
                        sstk.append("")
                        pstk.append(Paragraph(child.y0, child.x0, child.x0, child.x0, child.y0, child.y1, child.size, False, font_style_for(child.fontname), source_color(child)))
                if not cur_v:                                               # 文字入栈
                    if (                                                    # 根据当前字符修正段落属性
                        child.size > pstk[-1].size                          # 1. 当前字符比段落字体大
                        or len(sstk[-1].strip()) == 1                       # 2. 当前字符为段落第二个文字（考虑首字母放大的情况）
                    ) and child.get_text() != " ":                          # 3. 当前字符不是空格
                        pstk[-1].y -= child.size - pstk[-1].size            # 修正段落初始纵坐标，假设两个不同大小字符的上边界对齐
                        pstk[-1].size = child.size
                    sstk[-1] += child.get_text()
                else:                                                       # 公式入栈
                    if (                                                    # 根据公式左侧的文字修正公式的纵向偏移
                        not vstk                                            # 1. 当前字符是公式的第一个字符
                        and cls == xt_cls                                   # 2. 当前字符与前一个字符属于同一段落
                        and child.x0 > xt.x0                                # 3. 前一个字符在公式左侧
                    ):
                        vfix = child.y0 - xt.y0
                    vstk.append(child)
                # 更新段落边界，因为段落内换行之后可能是公式开头，所以要在外边处理
                pstk[-1].x0 = min(pstk[-1].x0, child.x0)
                pstk[-1].x1 = max(pstk[-1].x1, child.x1)
                pstk[-1].y0 = min(pstk[-1].y0, child.y0)
                pstk[-1].y1 = max(pstk[-1].y1, child.y1)
                # 更新上一个字符
                xt = child
                xt_cls = cls
            elif isinstance(child, LTFigure):   # 图表
                pass
            elif isinstance(child, LTLine):     # 线条
                layout = self.layout[ltpage.pageid]
                # ltpage.height 可能是 fig 里面的高度，这里统一用 layout.shape
                h, w = layout.shape
                # 读取当前线条在 layout 中的类别
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                if vstk and cls == xt_cls:      # 公式线条
                    vlstk.append(child)
                else:                           # 全局线条
                    lstk.append(child)
            else:
                pass
        # 处理结尾
        if vstk:    # 公式出栈
            sstk[-1] += f"{{v{len(var)}}}"
            var.append(vstk)
            varl.append(vlstk)
            varf.append(vfix)
        log.debug("\n==========[VSTACK]==========\n")
        for id, v in enumerate(var):  # 计算公式宽度
            l = max([vch.x1 for vch in v]) - v[0].x0
            log.debug(f'< {l:.1f} {v[0].x0:.1f} {v[0].y0:.1f} {v[0].cid} {v[0].fontname} {len(varl[id])} > v{id} = {"".join([ch.get_text() for ch in v])}')
            vlen.append(l)
        ############################################################
        # B. 段落翻译
        log.debug("\n==========[SSTACK]==========\n")

        latin_fonts = {
            "regular": "tiro",
            "bold": "tibo",
            "italic": "tiit",
            "bold_italic": "tibi",
        }

        def rendered_width(text: str, size: float, latin_font: str = "tiro") -> float:
            width = 0.0
            for char in text:
                try:
                    if self.fontmap[latin_font].to_unichr(ord(char)) == char:
                        width += self.fontmap[latin_font].char_width(ord(char)) * size
                    else:
                        width += self.noto.char_lengths(char, size)[0]
                except Exception:
                    width += size * 0.5
            return width

        @retry(wait=wait_fixed(1), stop=stop_after_attempt(3), reraise=True)
        def worker(item):  # 多线程翻译
            paragraph_id, s = item
            if not s.strip() or re.match(r"^\{v\d+\}$", s):  # 空白和公式不翻译
                return s
            try:
                # Machine programs and metadata are position-sensitive. Keep
                # commands/placeholders byte-for-byte so highlights and
                # aligned comments stay attached to the correct source row.
                if CODE_LINE_RE.match(s.strip()) or (
                    s.strip().startswith("{") and s.strip().endswith("}")
                ):
                    return s
                toc_entry = TOC_ENTRY_RE.match(s.strip())
                if toc_entry:
                    # Translate only the label. Rebuild the leader so the page
                    # number remains anchored at the original right boundary.
                    label = toc_entry.group("label").rstrip()
                    page_number = toc_entry.group("page").strip()
                    translated_label = self.translator.translate(label).strip()
                    paragraph = pstk[paragraph_id]
                    latin_font = latin_fonts[paragraph.font_style]
                    available = paragraph.x1 - paragraph.x0
                    fixed_width = rendered_width(
                        f"{translated_label}  {page_number}",
                        paragraph.size,
                        latin_font,
                    )
                    dot_width = max(
                        rendered_width(".", paragraph.size, latin_font), 0.1
                    )
                    dot_count = max(4, int((available - fixed_width) / dot_width))
                    new = f"{translated_label} {'.' * dot_count} {page_number}"
                else:
                    new = self.translator.translate(s)
                return new
            except BaseException as e:
                if log.isEnabledFor(logging.DEBUG):
                    log.exception(e)
                else:
                    log.exception(e, exc_info=False)
                raise e
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.thread
        ) as executor:
            news = list(executor.map(worker, enumerate(sstk)))

        def restore_source_breaks(text: str, paragraph_id: int) -> str:
            """Map explicit source line feeds to nearby target-language spaces."""
            break_positions = pstk[paragraph_id].break_positions
            source = sstk[paragraph_id]
            if not break_positions or not source or not text:
                return text

            # Translator-provided line endings are normalized first; the
            # source PDF remains the authority for the number of hard lines.
            normalized = re.sub(r"\s+", " ", text).strip()
            if not normalized:
                return text
            whitespace_positions = [
                match.start() for match in re.finditer(r" ", normalized)
            ]
            if not whitespace_positions:
                return normalized

            chosen: list[int] = []
            source_length = max(len(source), 1)
            for source_position in break_positions:
                target = round(source_position / source_length * len(normalized))
                candidates = [
                    position
                    for position in whitespace_positions
                    if position not in chosen
                    and (not chosen or position > chosen[-1])
                ]
                if not candidates:
                    break
                chosen.append(min(candidates, key=lambda position: abs(position - target)))

            for position in reversed(chosen):
                normalized = normalized[:position] + "\n" + normalized[position + 1:]
            return normalized

        def restore_placeholder_spacing(text: str) -> str:
            """Keep protected glyph runs separated from translated words."""
            marker = r"\{\s*v[\d\s]+\}"
            text = re.sub(rf"(?<=\w)(?={marker})", " ", text, flags=re.IGNORECASE)
            text = re.sub(
                rf"({marker})(?=\w)",
                r"\1 ",
                text,
                flags=re.IGNORECASE,
            )
            return text

        news = [
            restore_source_breaks(
                restore_placeholder_spacing(new),
                paragraph_id,
            )
            for paragraph_id, new in enumerate(news)
        ]

        ############################################################
        # C. 新文档排版
        def raw_string(fcur: str, cstk: str):  # 编码字符串
            if fcur == self.noto_name:
                return "".join(["%04x" % self.noto.has_glyph(ord(c)) for c in cstk])
            elif isinstance(self.fontmap[fcur], PDFCIDFont):  # 判断编码长度
                return "".join(["%04x" % ord(c) for c in cstk])
            else:
                return "".join(["%02x" % ord(c) for c in cstk])

        # 根据目标语言获取默认行距
        LANG_LINEHEIGHT_MAP = {
            "zh-cn": 1.4, "zh-tw": 1.4, "zh-hans": 1.4, "zh-hant": 1.4, "zh": 1.4,
            "ja": 1.1, "ko": 1.2, "en": 1.2, "ar": 1.0, "ru": 0.8, "uk": 0.8, "ta": 0.8
        }
        default_line_height = LANG_LINEHEIGHT_MAP.get(self.translator.lang_out.lower(), 1.1) # 小语种默认1.1
        _x, _y = 0, 0
        ops_list = []

        def color_operator(color) -> str:
            if color is None:
                return "0 g "
            try:
                components = tuple(float(component) for component in color)
            except (TypeError, ValueError):
                return "0 g "
            if len(components) == 1:
                return f"{components[0]:f} g "
            if len(components) == 3:
                return " ".join(f"{component:f}" for component in components) + " rg "
            if len(components) == 4:
                return " ".join(f"{component:f}" for component in components) + " k "
            return "0 g "

        def gen_op_txt(font, size, x, y, rtxt, color):
            return f"{color_operator(color)}/{font} {size:f} Tf 1 0 0 1 {x:f} {y:f} Tm [<{rtxt}>] TJ "

        def gen_op_line(x, y, xlen, ylen, linewidth):
            return f"ET q 1 0 0 1 {x:f} {y:f} cm [] 0 d 0 J {linewidth:f} w 0 0 m {xlen:f} {ylen:f} l S Q BT "

        for id, new in enumerate(news):
            x: float = pstk[id].x                       # 段落初始横坐标
            y: float = pstk[id].y                       # 段落初始纵坐标
            x0: float = pstk[id].x0                     # 段落左边界
            x1: float = pstk[id].x1                     # 段落右边界
            height: float = pstk[id].y1 - pstk[id].y0   # 段落高度
            size: float = pstk[id].size                 # 段落字体大小
            brk: bool = pstk[id].brk                    # 段落换行标记
            cstk: str = ""                              # 当前文字栈
            fcur: str = None                            # 当前字体 ID
            lidx = 0                                    # 记录换行次数
            tx = x
            fcur_ = fcur
            ptr = 0
            latin_font = latin_fonts[pstk[id].font_style]
            log.debug(f"< {y} {x} {x0} {x1} {size} {brk} > {sstk[id]} | {new}")

            inline_style_positions: dict[int, str] = {}
            for token, style in sorted(
                technical_bold_tokens.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            ):
                token_pattern = re.compile(
                    rf"(?<!\w){re.escape(token)}(?!\w)",
                    re.IGNORECASE,
                )
                for match in token_pattern.finditer(new):
                    for position in range(match.start(), match.end()):
                        inline_style_positions[position] = style

            ops_vals: list[dict] = []

            while ptr < len(new):
                if new[ptr] == "\n":
                    if cstk:
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": fcur,
                            "size": size,
                            "x": tx,
                            "dy": 0,
                            "rtxt": raw_string(fcur, cstk),
                            "lidx": lidx,
                            "color": pstk[id].color,
                        })
                        cstk = ""
                    ptr += 1
                    x = x0
                    lidx += 1
                    fcur = None
                    continue
                vy_regex = re.match(
                    r"\{\s*v([\d\s]+)\}", new[ptr:], re.IGNORECASE
                )  # 匹配 {vn} 公式标记
                mod = 0  # 文字修饰符
                if vy_regex:  # 加载公式
                    ptr += len(vy_regex.group(0))
                    try:
                        vid = int(vy_regex.group(1).replace(" ", ""))
                        adv = vlen[vid]
                    except Exception:
                        continue  # 翻译器可能会自动补个越界的公式标记
                    if var[vid][-1].get_text() and unicodedata.category(var[vid][-1].get_text()[0]) in ["Lm", "Mn", "Sk"]:  # 文字修饰符
                        mod = var[vid][-1].width
                else:  # 加载文字
                    ch = new[ptr]
                    active_latin_font = latin_fonts[
                        inline_style_positions.get(ptr, pstk[id].font_style)
                    ]
                    fcur_ = None
                    try:
                        if fcur_ is None and self.fontmap[active_latin_font].to_unichr(ord(ch)) == ch:
                            fcur_ = active_latin_font  # preserve source emphasis
                    except Exception:
                        pass
                    if fcur_ is None:
                        fcur_ = self.noto_name  # 默认非拉丁字体
                    if fcur_ == self.noto_name: # FIXME: change to CONST
                        adv = self.noto.char_lengths(ch, size)[0]
                    else:
                        adv = self.fontmap[fcur_].char_width(ord(ch)) * size
                    ptr += 1
                if (                                # 输出文字缓冲区
                    fcur_ != fcur                   # 1. 字体更新
                    or vy_regex                     # 2. 插入公式
                    or x + adv > x1 + 0.1 * size    # 3. 到达右边界（可能一整行都被符号化，这里需要考虑浮点误差）
                ):
                    if cstk:
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": fcur,
                            "size": size,
                            "x": tx,
                            "dy": 0,
                            "rtxt": raw_string(fcur, cstk),
                            "lidx": lidx,
                            "color": pstk[id].color,
                        })
                        cstk = ""
                if brk and x + adv > x1 + 0.1 * size:  # 到达右边界且原文段落存在换行
                    x = x0
                    lidx += 1
                if vy_regex:  # 插入公式
                    fix = 0
                    if fcur is not None:  # 段落内公式修正纵向偏移
                        fix = varf[vid]
                    for vch in var[vid]:  # 排版公式字符
                        vc = chr(vch.cid)
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": self.fontid[vch.font],
                            "size": vch.size,
                            "x": x + vch.x0 - var[vid][0].x0,
                            "dy": fix + vch.y0 - var[vid][0].y0,
                            "rtxt": raw_string(self.fontid[vch.font], vc),
                            "lidx": lidx,
                            "color": source_color(vch),
                        })
                        if log.isEnabledFor(logging.DEBUG):
                            lstk.append(LTLine(0.1, (_x, _y), (x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0)))
                            _x, _y = x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0
                    for l in varl[vid]:  # 排版公式线条
                        if l.linewidth < 5:  # hack 有的文档会用粗线条当图片背景
                            ops_vals.append({
                                "type": OpType.LINE,
                                "x": l.pts[0][0] + x - var[vid][0].x0,
                                "dy": l.pts[0][1] + fix - var[vid][0].y0,
                                "linewidth": l.linewidth,
                                "xlen": l.pts[1][0] - l.pts[0][0],
                                "ylen": l.pts[1][1] - l.pts[0][1],
                                "lidx": lidx
                            })
                else:  # 插入文字缓冲区
                    if not cstk:  # 单行开头
                        tx = x
                        if x == x0 and ch == " ":  # 消除段落换行空格
                            adv = 0
                        else:
                            cstk += ch
                    else:
                        cstk += ch
                adv -= mod # 文字修饰符
                fcur = fcur_
                x += adv
                if log.isEnabledFor(logging.DEBUG):
                    lstk.append(LTLine(0.1, (_x, _y), (x, y)))
                    _x, _y = x, y
            # 处理结尾
            if cstk:
                ops_vals.append({
                    "type": OpType.TEXT,
                    "font": fcur,
                    "size": size,
                    "x": tx,
                    "dy": 0,
                    "rtxt": raw_string(fcur, cstk),
                    "lidx": lidx,
                    "color": pstk[id].color,
                })

            line_height = default_line_height

            while (lidx + 1) * size * line_height > height and line_height >= 1:
                line_height -= 0.05

            for vals in ops_vals:
                if vals["type"] == OpType.TEXT:
                    ops_list.append(gen_op_txt(vals["font"], vals["size"], vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["rtxt"], vals["color"]))
                elif vals["type"] == OpType.LINE:
                    ops_list.append(gen_op_line(vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["xlen"], vals["ylen"], vals["linewidth"]))

        for l in lstk:  # 排版全局线条
            if l.linewidth < 5:  # hack 有的文档会用粗线条当图片背景
                ops_list.append(gen_op_line(l.pts[0][0], l.pts[0][1], l.pts[1][0] - l.pts[0][0], l.pts[1][1] - l.pts[0][1], l.linewidth))

        ops = f"BT {''.join(ops_list)}ET "
        return ops


class OpType(Enum):
    TEXT = "text"
    LINE = "line"
