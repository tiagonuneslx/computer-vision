# importar a biblioteca do opencv
import cv2

# ler uma imagem de teste (ajustar o nome e caminho da imagem a gosto
# e garantir que a imagem está no sítio certo
img = cv2.imread(r'test.jpg', 1)

# mostrar a imagem numa janela intitulada "Imagem de Teste".
cv2.imshow('Imagem de Teste', img)

# colocar em ciclo infinito, à espera que se carregue numa tecla
cv2.waitKey(0)

# destruir a janela criada
cv2.destroyAllWindows()
