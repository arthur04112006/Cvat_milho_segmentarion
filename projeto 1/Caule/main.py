from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

# ===== CONFIG =====
BASE_DIR = Path(__file__).resolve().parent
XML_FILE = BASE_DIR / "annotations.xml"
DEFAULT_IMAGE_FILE = BASE_DIR / "milho_01.jpg"
CM_PER_PIXEL = 0.0355
POLYLINE_MASK_THICKNESS = 3
POLYLINE_DRAW_THICKNESS = 3
# ==================


def parse_points(points_str):
    points = []
    for raw_point in points_str.split(";"):
        x_str, y_str = raw_point.split(",")
        points.append([int(float(x_str)), int(float(y_str))])
    return np.array(points, dtype=np.int32)


def resolve_image_file(root):
    image_tag = root.find("image")
    if image_tag is None:
        return DEFAULT_IMAGE_FILE

    xml_name = image_tag.attrib.get("name", "").strip()
    if xml_name:
        image_name = Path(xml_name).name
        candidates = [
            BASE_DIR / image_name,
            BASE_DIR.parent / image_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    if DEFAULT_IMAGE_FILE.exists():
        return DEFAULT_IMAGE_FILE

    jpg_files = sorted(BASE_DIR.glob("*.jpg"))
    if jpg_files:
        return jpg_files[0]

    return DEFAULT_IMAGE_FILE


tree = ET.parse(XML_FILE)
root = tree.getroot()
image_file = resolve_image_file(root)
image = cv2.imread(str(image_file))

if image is None:
    print(f"Erro: imagem não encontrada em: {image_file}")
    raise SystemExit(1)

xml_image_tag = root.find("image")
if xml_image_tag is not None:
    xml_width = int(xml_image_tag.attrib.get("width", image.shape[1]))
    xml_height = int(xml_image_tag.attrib.get("height", image.shape[0]))
    if (image.shape[1], image.shape[0]) != (xml_width, xml_height):
        image = cv2.resize(image, (xml_width, xml_height), interpolation=cv2.INTER_AREA)
        print(
            f"Imagem redimensionada de {image_file.name} para "
            f"{xml_width}x{xml_height} para coincidir com o XML."
        )

image_contornos = image.copy()
mascara = np.zeros(image.shape[:2], dtype=np.uint8)
segmentada = np.zeros_like(image)

contador = 0

for image_tag in root.findall("image"):
    anotacoes = list(image_tag.findall("polygon")) + list(image_tag.findall("polyline"))

    for anotacao in anotacoes:
        points_str = anotacao.attrib.get("points", "").strip()
        if not points_str:
            continue

        points = parse_points(points_str)
        if len(points) < 2:
            continue

        contador += 1
        label = anotacao.attrib.get("label", "objeto").capitalize()
        tipo = anotacao.tag

        if tipo == "polygon":
            cv2.fillPoly(mascara, [points], 255)
            cv2.polylines(image_contornos, [points], True, (0, 255, 0), POLYLINE_DRAW_THICKNESS)
            comprimento_pixels = max(cv2.boundingRect(points)[2:])
        else:
            cv2.polylines(
                mascara,
                [points],
                False,
                255,
                POLYLINE_MASK_THICKNESS,
                lineType=cv2.LINE_AA,
            )
            cv2.polylines(
                image_contornos,
                [points],
                False,
                (0, 255, 0),
                POLYLINE_DRAW_THICKNESS,
                lineType=cv2.LINE_AA,
            )
            comprimento_pixels = cv2.arcLength(points.astype(np.float32), False)

        x, y, w, h = cv2.boundingRect(points)
        comprimento_cm = comprimento_pixels * CM_PER_PIXEL

        print(f"{label} {contador}")
        print(f"Comprimento em pixels: {comprimento_pixels:.2f}")
        print(f"Comprimento em cm: {comprimento_cm:.2f}")
        print("-" * 30)

        cv2.putText(
            image_contornos,
            f"{label} {contador} - {comprimento_cm:.1f} cm",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

segmentada[mascara > 0] = image[mascara > 0]

print(f"Quantidade total de objetos: {contador}")

cv2.imwrite(str(BASE_DIR / "resultado_contornos.jpg"), image_contornos)
cv2.imwrite(str(BASE_DIR / "mascara_segmentacao.jpg"), mascara)
cv2.imwrite(str(BASE_DIR / "imagem_segmentada.jpg"), segmentada)

cv2.imshow("Imagem original com contornos", image_contornos)
cv2.imshow("Mascara da segmentacao", mascara)
cv2.imshow("Imagem segmentada", segmentada)

cv2.waitKey(0)
cv2.destroyAllWindows()
