import csv
from pathlib import Path
from openpyxl import Workbook
from t_rag.ingestion.readers import read_sections


def test_full_spreadsheet_and_csv(tmp_path):
    workbook = Workbook()
    for i in range(1, 45): workbook.active.append([i, f"value-{i}"])
    path = tmp_path / "table.xlsx"; workbook.save(path)
    sections = read_sections(path)
    assert "value-44" in sections[-1].text
    assert "行 31" in sections[-1].location
    path = tmp_path / "table.csv"
    with path.open("w") as stream:
        writer = csv.writer(stream)
        for i in range(44): writer.writerow([i, f"row-{i}"])
    assert "row-43" in read_sections(path)[-1].text


def test_markdown_heading_and_fenced_code(tmp_path):
    path = tmp_path / "a.mdc"
    path.write_text('---\ndescription: rule\n---\n# Root\nbody\n## Child\ntext\n```python\n# not a heading\n```')
    sections = read_sections(path)
    assert [s.location for s in sections] == ["Root", "Root / Child"]
    assert "not a heading" in sections[-1].text
    assert "description" not in sections[0].text


def test_pdf_all_pages(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    for number in range(1, 13):
        page = writer.add_blank_page(width=300, height=300)
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 200 Td (page-{number}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    path = tmp_path / "long.pdf"
    with path.open("wb") as output: writer.write(output)
    sections = read_sections(path)
    assert len(sections) == 12
    assert "page-11" in sections[10].text
    assert sections[10].location == "第 11 页"
