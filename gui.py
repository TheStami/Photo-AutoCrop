import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import threading
import queue
import cv2
from PIL import Image, ImageTk
import numpy as np

from autocrop import detect_boxes, crop_and_warp

class EditableBox:
    def __init__(self, canvas, points, scale, offset_x, offset_y, max_w, max_h, color="#00ff00"):
        self.canvas = canvas
        self.points = points.astype(np.float32)
        self.scale = scale
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.max_w = max_w
        self.max_h = max_h
        self.color = color
        self.handles = []
        self.polygon = None
        self.active_handle = None
        
        self.draw()

    def get_canvas_coords(self, pt):
        return pt[0] * self.scale + self.offset_x, pt[1] * self.scale + self.offset_y

    def get_img_coords(self, cx, cy):
        return (cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale

    def draw(self):
        flat_coords = []
        for p in self.points:
            cx, cy = self.get_canvas_coords(p)
            flat_coords.extend([cx, cy])
            
        if self.polygon is None:
            self.polygon = self.canvas.create_polygon(*flat_coords, outline=self.color, fill="", width=3)
        else:
            self.canvas.coords(self.polygon, *flat_coords)
        
        r = 7
        if not self.handles:
            for i, p in enumerate(self.points):
                cx, cy = self.get_canvas_coords(p)
                h = self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=self.color, outline="white", width=2)
                self.canvas.tag_bind(h, "<ButtonPress-1>", lambda e, idx=i: self.on_press(e, idx))
                self.canvas.tag_bind(h, "<B1-Motion>", self.on_drag)
                self.canvas.tag_bind(h, "<ButtonRelease-1>", self.on_release)
                self.handles.append(h)
        else:
            for i, p in enumerate(self.points):
                cx, cy = self.get_canvas_coords(p)
                self.canvas.coords(self.handles[i], cx-r, cy-r, cx+r, cy+r)

    def clear(self):
        if self.polygon:
            self.canvas.delete(self.polygon)
            self.polygon = None
        for h in self.handles:
            self.canvas.delete(h)
        self.handles = []

    def on_press(self, event, idx):
        self.active_handle = idx

    def on_drag(self, event):
        if self.active_handle is not None:
            ix, iy = self.get_img_coords(event.x, event.y)
            ix = max(0, min(self.max_w - 1, ix))
            iy = max(0, min(self.max_h - 1, iy))
            self.points[self.active_handle] = [ix, iy]
            self.draw()

    def on_release(self, event):
        self.active_handle = None


class AutoCropApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AutoCrop Pro - Edytor AI")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)
        
        self.input_dir = tk.StringVar(value=os.path.abspath("input"))
        self.output_dir = tk.StringVar(value=os.path.abspath("output"))
        
        self.queue = queue.Queue()
        self.files_data = {} 
        self.file_list = []
        
        self.current_filename = None
        self.current_boxes = []
        self.current_img = None
        self.current_tk_img = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.create_widgets()
        self.root.after(100, self.process_queue)
        
    def create_widgets(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # LEFT PANEL
        left_frame = ttk.Frame(main_pane, width=350)
        main_pane.add(left_frame, weight=1)
        
        ttk.Label(left_frame, text="Folder wejściowy (skany):").pack(anchor=tk.W, pady=(0, 2))
        ttk.Entry(left_frame, textvariable=self.input_dir, state="readonly").pack(fill=tk.X)
        ttk.Button(left_frame, text="Wybierz...", command=self.browse_input).pack(fill=tk.X, pady=(2, 10))
        
        ttk.Label(left_frame, text="Folder wyjściowy (pojedyncze zdjęcia):").pack(anchor=tk.W, pady=(0, 2))
        ttk.Entry(left_frame, textvariable=self.output_dir, state="readonly").pack(fill=tk.X)
        ttk.Button(left_frame, text="Wybierz...", command=self.browse_output).pack(fill=tk.X, pady=(2, 10))
        
        self.start_btn = ttk.Button(left_frame, text="Uruchom analizę AI w tle", command=self.start_processing)
        self.start_btn.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(left_frame, text="Zeskanowane pliki:").pack(anchor=tk.W)
        self.listbox = tk.Listbox(left_frame, font=("Courier", 10))
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.on_select_file)
        
        # Zapisz wszystko
        ttk.Button(left_frame, text="Zapisz WSZYSTKIE gotowe", command=self.save_all).pack(fill=tk.X, pady=(10, 0))
        
        # RIGHT PANEL
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=4)
        
        self.canvas = tk.Canvas(right_frame, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        toolbar = ttk.Frame(right_frame)
        toolbar.pack(fill=tk.X, pady=10)
        
        ttk.Button(toolbar, text="↶ Obróć zdjęcie w lewo", command=lambda: self.rotate_all(-1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Obróć zdjęcie w prawo ↷", command=lambda: self.rotate_all(1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="+ Dodaj brakujące zdjęcie", command=self.add_new_box).pack(side=tk.LEFT, padx=20)
        
        ttk.Button(toolbar, text="💾 Zapisz z tego skanu", command=self.save_current).pack(side=tk.RIGHT, padx=5)
        
        self.status_var = tk.StringVar(value="Gotowy. Kliknij 'Uruchom analizę AI w tle'.")
        ttk.Label(self.root, textvariable=self.status_var).pack(anchor=tk.W, padx=10, pady=5)
        
    def rotate_all(self, direction):
        if self.current_img is None: return
        
        h, w = self.current_img.shape[:2]
        
        # Fizyczny obrót całego obrazka załadowanego z dysku
        if direction == 1:
            self.current_img = cv2.rotate(self.current_img, cv2.ROTATE_90_CLOCKWISE)
            # Matematyczne dostosowanie punktów istniejących ramek
            for box in self.current_boxes:
                new_points = [[h - 1 - y, x] for (x, y) in box.points]
                box.points = np.array(new_points, dtype=np.float32)
        elif direction == -1:
            self.current_img = cv2.rotate(self.current_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Matematyczne dostosowanie punktów istniejących ramek
            for box in self.current_boxes:
                new_points = [[y, w - 1 - x] for (x, y) in box.points]
                box.points = np.array(new_points, dtype=np.float32)
                
        # Zapisz fizycznie obrócone zdjęcie w pamięci podręcznej
        if self.current_filename:
            self.files_data[self.current_filename]['img'] = self.current_img
            
        # Odrysuj canvas
        self.display_image(self.current_img)
        
        # Pobierz nowe wymiary PO obróceniu
        new_h, new_w = self.current_img.shape[:2]
        
        for box in self.current_boxes:
            box.max_w = new_w
            box.max_h = new_h
            box.scale = self.scale
            box.offset_x = self.offset_x
            box.offset_y = self.offset_y
            box.clear()
            box.draw()
            
        self.update_status("Obrócono całe zdjęcie (skan). Punkty zaktualizowały się automatycznie.")
            
    def add_new_box(self):
        if self.current_img is None: return
        h, w = self.current_img.shape[:2]
        cx, cy = w/2, h/2
        s = min(w, h) * 0.15
        pts = np.array([
            [cx-s, cy-s], [cx+s, cy-s], [cx+s, cy+s], [cx-s, cy+s]
        ])
        box = EditableBox(self.canvas, pts, self.scale, self.offset_x, self.offset_y, w, h, color="#ffaa00")
        self.current_boxes.append(box)
        self.update_status("Dodano nowy obszar. Przesuń punkty narożne.")

    def browse_input(self):
        folder = filedialog.askdirectory(initialdir=self.input_dir.get())
        if folder: self.input_dir.set(folder)

    def browse_output(self):
        folder = filedialog.askdirectory(initialdir=self.output_dir.get())
        if folder: self.output_dir.set(folder)

    def update_listbox(self):
        selection = self.listbox.curselection()
        self.listbox.delete(0, tk.END)
        for i, f in enumerate(self.file_list):
            status = self.files_data[f]['status']
            if status == "waiting": prefix = "[   ] "
            elif status == "processing": prefix = "[~AI] "
            else: prefix = "[OK ] "
            self.listbox.insert(tk.END, prefix + f)
            if selection and selection[0] == i:
                self.listbox.selection_set(i)

    def start_processing(self):
        in_dir = self.input_dir.get()
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        try:
            files = [f for f in os.listdir(in_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
        except Exception:
            self.update_status("Błąd: Folder wejściowy nie istnieje.")
            return
            
        if not files:
            self.update_status("Brak plików graficznych w folderze wejściowym.")
            return
            
        self.file_list = files
        self.files_data = {}
        for f in files:
            self.files_data[f] = {'status': 'waiting', 'img': None, 'boxes_data': []}
            
        self.update_listbox()
        self.start_btn.config(state=tk.DISABLED)
        self.update_status("Rozpoczęto analizę AI w tle. Możesz przeglądać gotowe wyniki.")
        
        threading.Thread(target=self.worker_thread, args=(in_dir, files), daemon=True).start()

    def worker_thread(self, in_dir, files):
        for f in files:
            self.queue.put(('status', f, 'processing'))
            path = os.path.join(in_dir, f)
            try:
                img, boxes = detect_boxes(path)
                self.queue.put(('done', f, img, boxes))
            except Exception as e:
                print(f"Błąd analizy {f}: {e}")
                img_fallback = cv2.imread(path)
                self.queue.put(('done', f, img_fallback, []))
                
        self.queue.put(('finish', None, None))

    def process_queue(self):
        while not self.queue.empty():
            msg = self.queue.get()
            msg_type = msg[0]
            if msg_type == 'status':
                self.files_data[msg[1]]['status'] = msg[2]
                self.update_listbox()
            elif msg_type == 'done':
                f, img, boxes = msg[1], msg[2], msg[3]
                self.files_data[f]['status'] = 'ready'
                self.files_data[f]['img'] = img
                self.files_data[f]['boxes_data'] = [b.copy() for b in boxes]
                self.update_listbox()
            elif msg_type == 'finish':
                self.start_btn.config(state=tk.NORMAL)
                messagebox.showinfo("Gotowe", "Analiza AI wszystkich skanów zakończona!")
                self.update_status("Analiza w tle zakończona.")
        self.root.after(200, self.process_queue)

    def on_select_file(self, event):
        selection = self.listbox.curselection()
        if not selection: return
        idx = selection[0]
        filename = self.file_list[idx]
        
        self.save_current_state_to_memory()
        
        data = self.files_data[filename]
        if data['status'] != 'ready':
            self.canvas.delete("all")
            self.current_filename = None
            for b in self.current_boxes: b.clear()
            self.current_boxes = []
            self.update_status("Plik wciąż jest przetwarzany przez AI. Czekaj...")
            return
            
        self.current_filename = filename
        self.current_img = data['img']
        self.update_status(f"Podgląd: {filename}. Przeciągaj kropki, aby dopasować.")
        
        self.root.update_idletasks()
        self.display_image(data['img'])
        
        for b in self.current_boxes:
            b.clear()
        self.current_boxes = []
        
        h_img, w_img = data['img'].shape[:2]
        for box_pts in data['boxes_data']:
            box = EditableBox(self.canvas, box_pts, self.scale, self.offset_x, self.offset_y, w_img, h_img)
            self.current_boxes.append(box)

    def save_current_state_to_memory(self):
        if self.current_filename and self.current_filename in self.files_data:
            data = self.files_data[self.current_filename]
            data['boxes_data'] = [b.points.copy() for b in self.current_boxes]

    def display_image(self, img):
        if img is None: return
        h, w = img.shape[:2]
        ch = self.canvas.winfo_height()
        cw = self.canvas.winfo_width()
        
        if ch < 10 or cw < 10:
            cw, ch = 800, 600 

        scale_w = cw / w
        scale_h = ch / h
        self.scale = min(scale_w, scale_h) * 0.95
        
        new_w = max(1, int(w * self.scale))
        new_h = max(1, int(h * self.scale))
        
        self.offset_x = (cw - new_w) / 2
        self.offset_y = (ch - new_h) / 2
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (new_w, new_h))
        self.current_tk_img = ImageTk.PhotoImage(image=Image.fromarray(resized))
        
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.current_tk_img)

    def _save_file(self, filename):
        data = self.files_data[filename]
        if data['status'] != 'ready': return 0
        
        out_dir = self.output_dir.get()
        if not os.path.exists(out_dir): os.makedirs(out_dir)
            
        name, ext = os.path.splitext(filename)
        count = 0
        for pts in data['boxes_data']:
            try:
                cropped = crop_and_warp(data['img'], np.float32(pts))
                out_filename = f"{name}_cropped_{count}{ext}" if count > 0 else f"{name}_cropped{ext}"
                out_path = os.path.join(out_dir, out_filename)
                cv2.imwrite(out_path, cropped)
                count += 1
            except Exception as e:
                print(f"Błąd podczas zapisywania obszaru {count} pliku {filename}: {e}")
        return count

    def save_current(self):
        if not self.current_filename: return
        self.save_current_state_to_memory()
        c = self._save_file(self.current_filename)
        self.update_status(f"Pomyślnie zapisano {c} zdjęć z pliku {self.current_filename}.")
        messagebox.showinfo("Zapisano", f"Zapisano {c} ujęć z tego skanu.")

    def save_all(self):
        self.save_current_state_to_memory()
        total = 0
        for f in self.file_list:
            total += self._save_file(f)
        self.update_status(f"Pomyślnie zapisano łącznie {total} zdjęć.")
        messagebox.showinfo("Zapisano", f"Zapisano łącznie {total} zdjęć ze wszystkich skanów.")

    def update_status(self, msg):
        self.status_var.set(msg)

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    app = AutoCropApp(root)
    root.mainloop()
