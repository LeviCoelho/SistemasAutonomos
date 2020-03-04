import cv2, os

# Funcao para busca de arquivos
def find(name, path):
    for root, dirs, files in os.walk(path):
        if (name in files) or (name in dirs):
            print("O diretorio/arquivo {} encontra-se em: {}".format(name, root))
            return os.path.join(root, name)
    # Caso nao encontre, recursao para diretorios anteriores
    return find(name, os.path.dirname(path))

# Importar arquivo XML
cv2path = os.path.dirname(cv2.__file__)
haar_path = find('haarcascades', cv2path)
xml_name = 'haarcascade_frontalface_alt2.xml'
xml_path = os.path.join(haar_path, xml_name)

# TODO: Inicializar Classificador
clf = cv2.CascadeClassifier(xml_path)

# Inicializar webcam
cap = cv2.VideoCapture(0)

# Loop para leitura do conteúdo
while(not cv2.waitKey(20) & 0xFF == ord('q')):
        # Capturar proximo frame
        ret, frame = cap.read()

        # TODO: Converter para tons de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TODO: Classificar
        faces = clf.detectMultiScale(gray)

        # TODO: Desenhar retangulo
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0))

        # Visualizar
        cv2.imshow('frame',frame)

# Desligar a webcam
cap.release()

#Fechar janela do vídeo
cv2.destroyAllWindows()