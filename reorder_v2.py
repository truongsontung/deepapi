#!/usr/bin/env python3
"""
Sắp xếp lại thứ tự các format trong _extract_xml_tags của tool_parser.py
và tạo tool_parser_dev.py với các format được đánh số và sắp xếp logic.
"""
import re
from pathlib import Path

def extract_blocks(lines):
    """Extract format blocks from lines of the function body."""
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith('# Format'):
            # Tìm số thứ tự
            match = re.search(r'# Format\s*(\d+[a-z]*)', line)
            if match:
                num = match.group(1)
                start = i
                # Tìm kết thúc: dòng tiếp theo bắt đầu bằng '# Format' hoặc dòng không thụt lề và không phải comment
                j = i + 1
                while j < len(lines):
                    if lines[j].strip().startswith('# Format'):
                        break
                    # Nếu dòng không thụt lề (không bắt đầu bằng 4 spaces hoặc tab) và không phải comment, coi là kết thúc
                    if not lines[j].startswith('    ') and not lines[j].startswith('\t') and lines[j].strip() != '' and not lines[j].strip().startswith('#'):
                        # Nhưng vì các dòng trong khối đều thụt lề, nên nếu gặp dòng không thụt lề (ngoài khối) thì dừng
                        break
                    j += 1
                end = j
                blocks.append((num, start, end))
                i = j
            else:
                i += 1
        else:
            i += 1
    return blocks

def main():
    src_path = Path('/home/vps2/deepapi/tool_parser.py')
    content = src_path.read_text()

    # Tìm vị trí hàm _extract_xml_tags
    start = content.find('def _extract_xml_tags(text: str) -> list:')
    if start == -1:
        raise ValueError("Không tìm thấy hàm")
    # Tìm kết thúc hàm: tìm 'def _extract_tool_calls_safe' hoặc 'def strip_tool_calls' hoặc 'def _get_valid_tool_set'
    end = content.find('def _extract_tool_calls_safe', start)
    if end == -1:
        end = content.find('def strip_tool_calls', start)
    if end == -1:
        end = content.find('def _get_valid_tool_set', start)
    if end == -1:
        raise ValueError("Không tìm thấy kết thúc hàm")

    before = content[:start]
    after = content[end:]

    # Lấy body của hàm (từ dòng '    """' đến hết)
    body = content[start:end]
    lines = body.splitlines(True)

    # Tìm dòng '    tools = []'
    tools_line = None
    for idx, line in enumerate(lines):
        if line.strip() == 'tools = []':
            tools_line = idx
            break
    if tools_line is None:
        raise ValueError("Không tìm thấy 'tools = []'")

    # Tìm dòng '    return tools'
    return_line = None
    for idx, line in enumerate(lines):
        if line.strip().startswith('return tools'):
            return_line = idx
            break
    if return_line is None:
        raise ValueError("Không tìm thấy 'return tools'")

    # Tách header, middle, footer
    header = ''.join(lines[:tools_line+1])  # bao gồm cả dòng 'tools = []'
    footer = ''.join(lines[return_line:])
    middle = ''.join(lines[tools_line+1:return_line])

    # Tách các khối format từ middle (dựa trên '# Format')
    middle_lines = middle.splitlines(True)
    blocks = extract_blocks(middle_lines)

    if not blocks:
        raise ValueError("Không tìm thấy khối format nào")

    print(f"Tìm thấy {len(blocks)} khối format")

    # Thứ tự ưu tiên mong muốn
    priority = [
        '26',
        '1', '1c',
        '2', '2b',
        '3',
        '6', '7',
        '5', '5b',
        '4',
        '8',
        '10', '10b', '10c',
        '11', '11b',
        '12',
        '13', '13b',
        '14', '14b',
        '15', '15b',
        '16', '16b',
        '17', '17b',
        '18', '18b',
        '19', '19b',
        '20', '20b',
        '21', '21b',
        '22', '23', '24', '25',
        '1b', '9',
        '27', '27b'
    ]

    # Xây dựng dict để dễ tra cứu
    block_dict = {num: (start, end) for num, start, end in blocks}

    # Sắp xếp các khối theo priority
    ordered = []
    for num in priority:
        if num in block_dict:
            ordered.append((num, block_dict[num]))
            del block_dict[num]
        else:
            print(f"Warning: Format {num} không tìm thấy")

    # Các khối còn lại (không có trong priority) thêm vào cuối
    for num, (start, end) in block_dict.items():
        ordered.append((num, (start, end)))

    # Xây dựng middle mới
    new_middle = ''
    for num, (start, end) in ordered:
        block = ''.join(middle_lines[start:end])
        new_middle += block
        new_middle += '\n'  # thêm dòng trống

    # Ghép lại body mới
    new_body = header + new_middle + footer

    # Toàn bộ file
    new_content = before + new_body + after

    # Ghi ra file
    out_path = Path('/home/vps2/deepapi/tool_parser_dev.py')
    out_path.write_text(new_content)
    print(f"Đã tạo {out_path}")

if __name__ == '__main__':
    main()
