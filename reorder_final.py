#!/usr/bin/env python3
import re
from pathlib import Path

content = Path('/home/vps2/deepapi/tool_parser.py').read_text()

# Tìm vị trí hàm _extract_xml_tags
start = content.find('def _extract_xml_tags(text: str) -> list:')
if start == -1:
    raise ValueError("Không tìm thấy hàm")

end = content.find('def _extract_tool_calls_safe', start)
if end == -1:
    end = content.find('def strip_tool_calls', start)
if end == -1:
    end = content.find('def _get_valid_tool_set', start)
if end == -1:
    raise ValueError("Không tìm thấy kết thúc hàm")

before = content[:start]
after = content[end:]

body = content[start:end]
lines = body.splitlines(True)

# Tìm dòng 'tools = []' và 'return tools'
tools_line = None
return_line = None
for i, line in enumerate(lines):
    if line.strip() == 'tools = []':
        tools_line = i
    if line.strip().startswith('return tools'):
        return_line = i

if tools_line is None or return_line is None:
    raise ValueError("Không tìm thấy tools = [] hoặc return tools")

header = ''.join(lines[:tools_line+1])
footer = ''.join(lines[return_line:])
middle_lines = lines[tools_line+1:return_line]

# Tìm các khối format
blocks = []
i = 0
while i < len(middle_lines):
    line = middle_lines[i]
    stripped = line.strip()
    if stripped.startswith('# Format'):
        match = re.search(r'# Format\s*(\d+[a-z]*)', line)
        if match:
            num = match.group(1)
            start_idx = i
            j = i + 1
            # Tìm kết thúc khối: gặp comment '# Format' khác, hoặc dòng không thụt lề
            while j < len(middle_lines):
                if middle_lines[j].strip().startswith('# Format'):
                    break
                if not middle_lines[j].startswith('    ') and not middle_lines[j].startswith('\t') and middle_lines[j].strip() != '':
                    break
                j += 1
            end_idx = j
            blocks.append((num, start_idx, end_idx))
            i = j
        else:
            i += 1
    else:
        i += 1

print(f"Tìm thấy {len(blocks)} khối format")

# Thứ tự ưu tiên
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

block_dict = {num: (s, e) for num, s, e in blocks}
ordered = []
for num in priority:
    if num in block_dict:
        ordered.append((num, block_dict[num]))
        del block_dict[num]

# Các khối còn lại thêm vào cuối
for num, (s, e) in block_dict.items():
    ordered.append((num, (s, e)))

# Tạo middle mới
new_middle = ''
for num, (s, e) in ordered:
    block = ''.join(middle_lines[s:e])
    new_middle += block + '\n'

new_body = header + new_middle + footer
new_content = before + new_body + after

out = Path('/home/vps2/deepapi/tool_parser_dev.py')
out.write_text(new_content)
print(f"Đã tạo {out}")
