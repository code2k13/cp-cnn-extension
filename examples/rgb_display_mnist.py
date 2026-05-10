#A memory-efficient CNN implementation using cnn_helper to perform digit recognition and display results on an ST7735R LCD.

import array
import gc
import math
import cnn_helper
import gc
import board
import busio
import displayio
import fourwire
import terminalio
from adafruit_st7735r import ST7735R
from adafruit_display_text import label
import time

displayio.release_displays()

spi = busio.SPI(
            clock=board.GP10,
            MOSI=board.GP11,
        )

display_bus = fourwire.FourWire(
            spi,
            command=board.GP16,
            chip_select=board.GP18,
            reset=board.GP17,
        )

display = ST7735R(
            display_bus,
            width=128,
            height=160,
            bgr=True,
        )

display.rotation = 0
display.auto_refresh = False

#Needed to overcome memory curroption
def show_result(image_path, prediction, confidence, inference_ms):

    try:

        root = displayio.Group()
        IMG_SIZE = 30
        bitmap = displayio.Bitmap(IMG_SIZE, IMG_SIZE, 65536)

        with open(image_path, "rb") as f:
            for y in range(IMG_SIZE):
                for x in range(IMG_SIZE):

                    lo = f.read(1)[0]
                    hi = f.read(1)[0]

                    color = (hi << 8) | lo

                    bitmap[x, y] = color

        tilegrid = displayio.TileGrid(
            bitmap,
            pixel_shader=displayio.ColorConverter(
                input_colorspace=displayio.Colorspace.RGB565_SWAPPED
            ),
            x=50,
            y=25,
        )

        root.append(tilegrid)

        pred_label = label.Label(
            terminalio.FONT,
            text="Pred: {}".format(prediction),
            color=0xFFFFFF,
            x=4,
            y=90,
        )

        conf_label = label.Label(
            terminalio.FONT,
            text="Conf: {:.1f}%".format(confidence),
            color=0xFFFF00,
            x=4,
            y=110,
        )

        time_label = label.Label(
            terminalio.FONT,
            text="Time: {} ms".format(inference_ms),
            color=0x00FF00,
            x=4,
            y=130,
        )

        root.append(pred_label)
        root.append(conf_label)
        root.append(time_label)     

        display.root_group = root
        display.refresh()

    except Exception as e:
        print("DISPLAY ERROR:", e)

    try:
        display.root_group = None
    except:
        pass

    gc.collect()



INPUT_SIZE = 900 
inp = array.array("f", [0.0] * INPUT_SIZE)

def simple_softmax(scores):
    shift_scores = [math.exp(s - max(scores)) for s in scores]
    sum_exp = sum(shift_scores)
    return [s / sum_exp for s in shift_scores]

def process_sample(file_path):
    try:
        with open(file_path, "rb") as f:
            f.readinto(inp)
            
        cnn_helper.load_model("model.bin")
        out = cnn_helper.perform_inference(inp)
        cnn_helper.unload_model()
        gc.collect()
        
        return out
    except Exception as e:
        print("Error:", e)
        return None
    
while True:
    for i in range(20):
        filename = "mnist_samples/sample_{:02d}.bin".format(i)
        image_path  =  "mnist_samples/sample_{:02d}.rgb".format(i)
        start_time = time.monotonic_ns()
        raw_output = process_sample(filename)
        
        if raw_output:
            probs = simple_softmax(raw_output)
            end_time = time.monotonic_ns()
            prediction = probs.index(max(probs))
            confidence = max(probs) * 100
            
            print("File: {} -> Pred: {} ({:.1f}%)".format(filename, prediction, confidence))
            show_result(image_path, prediction, confidence, (end_time-start_time)/1_000_000)
        time.sleep(2)
        gc.collect()
