import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

# ===== CONFIG =====
BASE_DIR = Path(__file__).resolve().parent
XML_FILE = BASE_DIR / "annotations.xml"
IMAGE_FILE = BASE_DIR / "milho_01.jpg"

CM_PER_PIXEL = 0.0355
ESPESSURA_MASCARA = 35
# ==================

image = cv2.imread(str(IMAGE_FILE))

if image is None:
    print("Erro: imagem nao encontrada. Verifique o nome do arquivo.")
    exit()

# Copias para saida
image_contornos = image.copy()
mascara = np.zeros(image.shape[:2], dtype=np.uint8)
segmentada = np.zeros_like(image)

tree = ET.parse(str(XML_FILE))
root = tree.getroot()

contador = 0

for image_tag in root.findall("image"):

    xml_width = int(image_tag.attrib.get("width", image.shape[1]))
    xml_height = int(image_tag.attrib.get("height", image.shape[0]))

    if image.shape[1] != xml_width or image.shape[0] != xml_height:
        image = cv2.resize(image, (xml_width, xml_height), interpolation=cv2.INTER_AREA)
        image_contornos = image.copy()
        mascara = np.zeros(image.shape[:2], dtype=np.uint8)
        segmentada = np.zeros_like(image)

    for polyline in image_tag.findall("polyline"):

        contador += 1

        points_str = polyline.attrib["points"]
        points = []

        for p in points_str.split(";"):
            x, y = map(float, p.split(","))
            points.append([int(x), int(y)])

        points = np.array(points, dtype=np.int32)

        # Desenhar a mascara do caule com espessura para gerar area segmentada
        cv2.polylines(mascara, [points], False, 255, ESPESSURA_MASCARA, lineType=cv2.LINE_8)

        # Desenhar contorno na imagem original
        cv2.polylines(image_contornos, [points], False, (0, 255, 0), 3)

        x, y, w, h = cv2.boundingRect(points)

        comprimento_pixels = cv2.arcLength(points.astype(np.float32), False)
        comprimento_cm = comprimento_pixels * CM_PER_PIXEL

        print(f"Caule {contador}")
        print(f"Comprimento em pixels: {comprimento_pixels:.2f}")
        print(f"Comprimento em cm: {comprimento_cm:.2f}")
        print("-" * 30)

        cv2.putText(
            image_contornos,
            f"Caule {contador} - {comprimento_cm:.1f} cm",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

# Aplicar mascara na imagem original
segmentada[mascara > 0] = image[mascara > 0]

print(f"Quantidade total de caules: {contador}")

# Salvar resultados
cv2.imwrite(str(BASE_DIR / "resultado_contornos.jpg"), image_contornos)
cv2.imwrite(str(BASE_DIR / "mascara_segmentacao.jpg"), mascara)
cv2.imwrite(str(BASE_DIR / "imagem_segmentada.jpg"), segmentada)

# Mostrar resultados
cv2.imshow("Imagem original com contornos", image_contornos)
cv2.imshow("Mascara da segmentacao", mascara)
cv2.imshow("Imagem segmentada", segmentada)

cv2.waitKey(0)
cv2.destroyAllWindows()
