# Realtime-audio2audio-alignment

This repository contains the real-time audio to audio alignment program which tracks the coorsponding audio time in the refence audio with incoming audio stream. The program is implemented by python 2.7 and Librosa with multi-core technology on Windows platform.

## Prerequisites
- Windows platform (Linux & OSX is not guaranteed)
- Python 2.7 
- Numpy
- Librosa ([here](https://librosa.github.io/))
- Tkinter ([here](https://docs.python.org/3/library/tk.html))
- PyAudio ([here](http://people.csail.mit.edu/hubert/pyaudio/))
- dill ([here](https://github.com/uqfoundation/dill))
- ast ([here](https://docs.python.org/2/library/ast.html))


## Run the program
- Connect the audio output of your computer to both speaker and audio interface input simultaneously.
- Run the program by:
```bash
python main.py
```
- Press "p" key on GUI window to start realtime alignment process
- check alignment result file "alignment_result.txt" under same directory with main.py

## Demo Video
- Demo video is available on youtube: [here](https://youtu.be/04JtjkpsU_0)


## License

The audio and annotation files are published under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
