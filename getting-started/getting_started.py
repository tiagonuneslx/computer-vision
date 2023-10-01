# importar a biblioteca do opencv
import cv2

# ler uma imagem de teste (ajustar o nome e caminho da imagem a gosto
# e garantir que a imagem está no sítio certo
img = cv2.imread(r'test.jpg', cv2.IMREAD_UNCHANGED)
imgGray = cv2.imread(r'test.jpg', cv2.IMREAD_GRAYSCALE)

print(f"ndim: {img.ndim}")
print(f"shape: {img.shape}")
print(f"size: {img.size}")
print(f"dtype: {img.dtype}")
print(f"itemsize: {img.itemsize}")

imgRgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
imgSquare = cv2.resize(img, (2000, 2000), cv2.INTER_LANCZOS4)
imgCropped = img[200:700, 200:700]

(t, mask) = cv2.threshold(imgGray, 100, 255, cv2.THRESH_BINARY_INV)

maskedImg = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("img", imgCropped)
cv2.waitKey(0)

cv2.imwrite('test_greyscale.jpg', imgGray)

# destruir a janela criada
cv2.destroyAllWindows()
