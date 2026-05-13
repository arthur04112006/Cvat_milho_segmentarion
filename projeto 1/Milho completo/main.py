import cv2
import numpy as np
import xml.etree.ElementTree as ET

# ===== CONFIG =====
XML_FILE = "annotations.xml"
IMAGE_FILE = "milho_01.jpg"

CM_PER_PIXEL = 0.0355
# ==================

image = cv2.imread(IMAGE_FILE)

if image is None:
    print("Erro: imagem não encontrada. Verifique o nome do arquivo.")
    exit()

# Cópias para saída
image_contornos = image.copy()
mascara = np.zeros(image.shape[:2], dtype=np.uint8)
segmentada = np.zeros_like(image)

tree = ET.parse(XML_FILE)
root = tree.getroot()

contador = 0

for image_tag in root.findall("image"):

    for polygon in image_tag.findall("polygon"):

        contador += 1

        points_str = polygon.attrib["points"]
        points = []

        for p in points_str.split(";"):
            x, y = map(float, p.split(","))
            points.append([int(x), int(y)])

        points = np.array(points, dtype=np.int32)

        # Preencher a máscara do milho
        cv2.fillPoly(mascara, [points], 255)

        # Desenhar contorno na imagem original
        cv2.polylines(image_contornos, [points], True, (0, 255, 0), 3)

        x, y, w, h = cv2.boundingRect(points)

        comprimento_pixels = max(w, h)
        comprimento_cm = comprimento_pixels * CM_PER_PIXEL

        print(f"Milho {contador}")
        print(f"Comprimento em pixels: {comprimento_pixels}")
        print(f"Comprimento em cm: {comprimento_cm:.2f}")
        print("-" * 30)

        cv2.putText(
            image_contornos,
            f"Milho {contador} - {comprimento_cm:.1f} cm",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

# Aplicar máscara na imagem original
segmentada[mascara == 255] = image[mascara == 255]

print(f"Quantidade total de milhos: {contador}")

# Salvar resultados
cv2.imwrite("resultado_contornos.jpg", image_contornos)
cv2.imwrite("mascara_segmentacao.jpg", mascara)
cv2.imwrite("imagem_segmentada.jpg", segmentada)

# Mostrar resultados
cv2.imshow("Imagem original com contornos", image_contornos)
cv2.imshow("Mascara da segmentacao", mascara)
cv2.imshow("Imagem segmentada", segmentada)

cv2.waitKey(0)
cv2.destroyAllWindows()