import cv2
import glob
import numpy as np
from scipy.stats import chi2
from plot_ellipse import plot_ellipse

class Kalman:
    """Kalman filter class.
    Creates a parameterized Kalman Filter instance.

    """
    def __init__(self, A: np.array, B: np.array, C: np.array, R: np.array, Q: np.array):
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q

    def predict(self, state: np.array, state_cov: np.array) -> (np.array, np.array):
        """Calculate de prediction based on state and covariance matrix

        Parameters
        ----------
        state : np.array
            Last state
        state_cov : last covariance matrix
            Last covariance matrix

        Returns
        -------
        list
            a list of containing two np.array, the predicted state and the predicted covariance matrix
        """
        return [self.A@state, self.A@state_cov@self.A.T + self.Q]
    
    def update(self, state, state_cov, state_pred, state_cov_pred, observation):

        S = self.C@state_cov_pred@self.C.T+self.R

        K = state_cov_pred@self.C.T@np.linalg.inv(S)

        r = observation[0:2,:] - self.C@state_pred
        # Update
        return [state_pred + K@r,
                (np.eye(4,4)-K@self.C)@state_cov_pred]
    
class Detection:

    def __init__(self, name, kalman):
        self.name = name
        self.state_history = []
        self.observation_history = []
        self.missed_observations_limit = 10
        self.missed_observations = 0
        self.last_observation_frame = 0
        self.kalman = kalman
        self.state =  None
        self.state_cov = None
        self.state_pred = None
        self.state_cov_pred = None
        self.initialized = False

    def addObservation(self, frame, observation):
        self.last_observation_frame = frame
        self.observation_history.append(observation)
        self.missed_observations = 0

        if self.initialized:
            [self.state, self.state_cov] = self.kalman.update(self.state, self.state_cov, self.state_pred, self.state_cov_pred, observation)
        else:
            self.initialized = True
            self.state = observation
            self.state_cov = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])     
        self.state_history.append(self.state)

    def addMissedObservation(self, frame):
        self.missed_observations = self.missed_observations + 1
        self.state = self.state_pred
        self.state_cov = self.state_cov_pred
        self.state_history.append(self.state)
        if self.missed_observations >= self.missed_observations_limit:
            return False
        else:
            return True
        
    def predict(self):
        [self.state_pred, self.state_cov_pred] = self.kalman.predict(self.state, self.state_cov)

    def mark(self, image, x, y, w, h):
        # cv2.rectangle(image, 
        #     (x, y),
        #     (x+w, y+h),
        #     (255,255,255), 4)
        marker = (int(x+(w/2)), int(y+(h/2)))
        cv2.drawMarker(image, marker, color=(0,0,255))
        cv2.putText(image,self.name, np.array(marker)-5,cv2.FONT_HERSHEY_PLAIN,1, (0,255,0), 2)

        plot_ellipse(image,self.state[0:2,:],self.state_cov[0:2,0:2],color=(255,0,0), label=f"Update {self.name}")
        self.plot_predict_ellipse(image)
    
    def plot_predict_ellipse(self, image):
        if self.state_pred is not None and self.state_cov_pred is not None:
            plot_ellipse(image,self.state_pred[0:2,:],self.state_cov_pred[0:2,0:2],color=(0,255,0), label=f"Prediction {self.name}")
            
def non_maxima_suppression(boxes, probabilities, overlap_threshold):
    """
    Aplica supresión no máxima a un conjunto de ventanas/cajas correspondientes a diferentes detecciones. Cada ventana
    tiene una probabilidad asociada.
    :param boxes: Lista de ventanas/cajas de detecciones para un mismo objeto.
    :param probabilities: Lista de probabilidades asociadas a cada ventana/caja.
    :param overlap_threshold: Umbral de solapamiento usado para determinar cuáles ventanas suprimir.
    :return: Una lista suprimida de detecciones.
    """
    # Si no hay cajas, no hay nada que hacer.
    if len(boxes) == 0:
        return [],[]
 
    # Si las cajas son enteras, tenemos que convertirlas a floats, puesto que efectuaremos muchas divisiones
    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')
 
    pick = []  # Contendrá los índices de las cajas que retornaremos.
 
    # Extraemos las coordenadas de las cajas.
    x_1 = boxes[:, 0]
    y_1 = boxes[:, 1]
    x_2 = x_1+boxes[:, 2]
    y_2 = y_1+boxes[:, 3]
 
    # Calculamos el área de las cajas, y las ordenamos ascendentemente por su probabilidad.
    area = (x_2 - x_1) * (y_2 - y_1)
    indexes = np.argsort(probabilities)
 

    # Iteramos sobre los índices
    while len(indexes) > 0:
        # Tomamos el último índice en la lista y lo añadimos a la lista de cajas seleccionadas.
        last = len(indexes)-1
        i = indexes[last]
        pick.append(i)
 
        # Encontramos las coordenadas (x, y) más grandes para el principio de la ventana, y los (x, y) más pequeños para
        # el final

        xx_1 = np.maximum(x_1[i], x_1[indexes[:last]])
        yy_1 = np.maximum(y_1[i], y_1[indexes[:last]])
        xx_2 = np.minimum(x_2[i], x_2[indexes[:last]])
        yy_2 = np.minimum(y_2[i], y_2[indexes[:last]])

        # Calculamos el ancho y la altura de la caja.
        width = np.maximum(0, (xx_2 - xx_1))
        height = np.maximum(0, (yy_2 - yy_1))
        
        intersection = width*height
        union = area[i] + area[indexes[:last]] - intersection
        # Calculamos la tasa de solapamiento.
        overlap = intersection / union
        # print(f"Overlaps: {overlap}")
        # print(f"Indexes: {indexes}")
 
        # Borramos todos los índices correspondientes a cajas con un solapamiento mayor al umbral.
        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
 
    return boxes[pick].astype('int'), probabilities[pick]  # Retornamos las cajas sobrevivientes, no sin antes convertirlas a enteros.

# INITIALIZATION
# --------------------------------------------------------------
# Initialize your parameters, HOG detector, Kalman filter, etc


# Load dataset
# --------------------------------------------------------------
# Change the path to the dataset
# files = glob.glob('./dataset/Data/TestData/2012-04-02_120351/RectGrabber/imgrect_000*_c0.pgm')
# files = glob.glob('./dataset/PETS09/output_*.jpg')
# files = glob.glob('./dataset/couple_walking/output_*.jpg')
files = glob.glob('./dataset/winter_pedestrians/output_*.jpg')
# files = glob.glob('./dataset/town_centre/output_*.jpg')

# files = glob.glob('./dataset/pedestrian/output_*.jpg')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# We assume that the filesnames are ordered alphabetically/numerically
deltaT =1

A = np.array([[1, 0, deltaT, 0     ],
              [0, 1, 0     , deltaT],
              [0, 0, 1     , 0     ], 
              [0, 0, 0     , 1     ]]) # Matriz de transformación de estado

B = np.array([[deltaT, 0     ],
              [0     , deltaT],
              [deltaT, 0     ],
              [0     , deltaT]])

C = np.array([[1, 0, 0, 0], 
              [0, 1, 0, 0]]) # Matriz de estado a medida

# Matriz de covarianzas de medidas (deltas)
# Incertidumbre en pixels
R = np.array([[100,0],
              [0,100]]) 

# Matriz de covarianzas del proceso (ruido dinámico, de velocidad) (epsilons)
                    # Me fio de mi predicción
Q = B@np.array([[0.7, 0],
                    [0, 0.7]])@B.T 

names = list([chr(i) for i in range(65,90)])

kalman = Kalman(A, B, C, R, Q)

initialized = False

current_detections = []
next_detections = []

for f, filename in enumerate(sorted(files)):
    # We load the current image
    image = cv2.imread(filename)

    # Detect a person using HOGDescriptor
    # Draw rectangles in the image around each detected object
    # detectMultiScale returns a vector of rectangles and scores
    #
    #  -each rectangle is defined as a tuple with the top-left corner,
    #   width and height.
    #  -each score is the classifier confidence for the corresponding
    #   rectangle

    # Image desired width
    rw = 640
    (h, w) = image.shape[:2] 
    r = rw / float(w) # ratio
    # Dimension maintain aspect-ratio
    dim = (rw, int(h * r)) 

    # Resized image
    resized =  cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # Detection of boxes
    rects, weights = hog.detectMultiScale(
        resized,
        winStride=(4, 4),
        padding=(8,8),
        scale=1)
        
    # Apply non maxima supression and draw the red rectangle
    rects, weights = non_maxima_suppression(np.array(rects), weights, 0.4)
    for i, rect in enumerate(rects):
        (x, y, w, h) = np.multiply(rect,1/r).astype(int)
        cv2.putText(image, str(i), (x,y),cv2.FONT_HERSHEY_PLAIN,1, (0,0,0), 2)
        cv2.rectangle(image, 
            (x, y),
            (x+w, y+h),
            (255,0,0), 2)

    # Calculate predictions
    for detection in current_detections:
        detection.predict()
        
    # For each detection check Mahalanobis distance or add new detecion
    print(f"Rects: {len(rects)}")
    print(f"Current:  {len(current_detections)}")

    # Itete over selected rects
    for rect,weight in zip(rects,weights):
        # Return rect to image dimensions
        (x, y, w, h) = np.multiply(rect,1/r).astype(int)

        # If detection above confidence
        if weight > 0.6:  

            ## Mark the confident detection in green
            cv2.rectangle(image, 
            (x, y),
            (x+w, y+h),
            (0,255,0), 2)

            # Get the center of the rect
            marker = (int(x+(w/2)), int(y+(h/2)))       
            # Set the candidate    
            candidate = np.array([[marker[0],marker[1],0,0]]).T

            # Get the threshold
            threshold = chi2.ppf(0.95, 2)
            candidate_match = None
            min_distance = None

            # Search nearest candidate within current detections
            for i, detection in enumerate(current_detections):
                # Get state difference
                state_diff = candidate[0:2,:]-detection.state_pred[0:2,:]
                # Calculate mahalanobis_distance
                mah_distance = state_diff.T@np.linalg.inv(detection.state_cov_pred[0:2,0:2])@state_diff
                if mah_distance < threshold and (min_distance is None or mah_distance < min_distance):
                    min_distance = mah_distance
                    candidate_match = i

            #If no match, add new detection else ,
            # move from current_detections to next_detections
            if candidate_match is None:
                new_detection = Detection(names.pop(),kalman)
                new_detection.addObservation(f, candidate)
                next_detections.append(new_detection)
                new_detection.mark(image, x, y, w, h)
            else:
                detection = current_detections.pop(candidate_match)
                detection.addObservation(f, candidate)   
                detection.mark(image, x, y, w, h)      
                next_detections.append(detection)

    # Missed observation will be added to detections not matched
    for i, detection in enumerate(current_detections):
        # Iterate over current not detecteion and predict
        missed_detection = current_detections.pop(i)
        if missed_detection.addMissedObservation(f) :
            next_detections.append(missed_detection)
            missed_detection.plot_predict_ellipse(image)
        else:
            names.append(missed_detection.name)
    
    current_detections = next_detections
    next_detections = []

    # Show the final image
    cv2.imshow("Multi tracking kalman filter", image)

    # We show the image for 10 ms
    # # and stop the loop if any key is pressed
    k = cv2.waitKey()
    if k==27:    # Esc key to stop
        break

    # if k != -1:
    #     break