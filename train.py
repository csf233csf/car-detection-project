from ultralytics import YOLO

class Train(object):
    def __init__(self, data, epoch, batch, project) -> None:
        # using a pretained yolo model
        self.model = YOLO('yolov8n.pt')
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
        data = 'data.yaml',
        epoch = 200, 
        batch = 0.5, 
        project	= 'checkpoint'                
    )