from ultralytics import YOLO

class Train(object):
    def __init__(self, model_path,  data, epoch, batch, project) -> None:
        # using a pretained yolo model
        self.model = YOLO(model_path)
        self.model.train(
            data = data,
            epochs = epoch,
            batch = batch,
            imgsz = 640,
            project = project,
            plots = True
        )

if __name__ == '__main__':
    INST = Train(
        model_path = 'checkpoint/train/weights/last.pt',
        data = 'data.yaml',
        epoch = 100, 
        batch = 0.5, 
        project	= 'checkpoint'                
    )
    