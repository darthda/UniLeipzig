import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialisiere Hauptfenster der App
class RedditClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reddit Kommentar-Klassifizierer")
        self.root.geometry("1000x800")  # Größeres Fenster für ansprechendes Design
        self.comments = []  # Liste für Kommentare
        self.results = pd.DataFrame(columns=['Comment', 'Max', 'Jakob', 'Kai', 'Eliano', 'Maria', 'Natalya'])

        # GUI-Elemente erstellen
        self.create_widgets()

    def create_widgets(self):
        # Titel
        title_label = tk.Label(
            self.root,
            text="Reddit Kommentar-Klassifizierer",
            font=("Helvetica", 28, "bold"),
            fg="white",
            bg="darkblue",
            padx=20,
            pady=10
        )
        title_label.pack(fill="x")

        # Buttons mit ansprechendem Design
        button_frame = tk.Frame(self.root, bg="lightgray")
        button_frame.pack(pady=20)

        self.load_button = tk.Button(
            button_frame,
            text="Kommentare laden",
            font=("Helvetica", 16),
            bg="deepskyblue",
            fg="white",
            width=20,
            command=self.load_comments
        )
        self.load_button.grid(row=0, column=0, padx=10, pady=10)

        self.classify_button = tk.Button(
            button_frame,
            text="Kommentare klassifizieren",
            font=("Helvetica", 16),
            bg="mediumseagreen",
            fg="white",
            width=20,
            command=self.classify_comments
        )
        self.classify_button.grid(row=0, column=1, padx=10, pady=10)

        self.visualize_button = tk.Button(
            button_frame,
            text="Ergebnisse anzeigen",
            font=("Helvetica", 16),
            bg="tomato",
            fg="white",
            width=20,
            command=self.visualize_results
        )
        self.visualize_button.grid(row=0, column=2, padx=10, pady=10)

        self.save_button = tk.Button(
            button_frame,
            text="Ergebnisse speichern",
            font=("Helvetica", 16),
            bg="gold",
            fg="black",
            width=20,
            command=self.save_results
        )
        self.save_button.grid(row=0, column=3, padx=10, pady=10)

        # Fußzeile
        footer_label = tk.Label(
            self.root,
            text="Entwickelt von Team Data Science",
            font=("Helvetica", 14, "italic"),
            fg="darkgray",
            bg="lightgray"
        )
        footer_label.pack(side="bottom", fill="x", pady=10)

    def load_comments(self):
        # Lade Datei mit Kommentaren
        file_path = filedialog.askopenfilename(filetypes=[("Textdateien", "*.txt"), ("CSV-Dateien", "*.csv")])
        if not file_path:
            return

        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                self.comments = [line.strip() for line in file.readlines() if line.strip()]
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            if 'Comment' in df.columns:
                self.comments = df['Comment'].dropna().tolist()

        self.results = pd.DataFrame(self.comments, columns=['Comment'])
        for name in ['Max', 'Jakob', 'Kai', 'Eliano', 'Maria', 'Natalya']:
            self.results[name] = None

        messagebox.showinfo("Erfolg", f"{len(self.comments)} Kommentare geladen!")

    def classify_comments(self):
        if not self.comments:
            messagebox.showerror("Fehler", "Keine Kommentare geladen!")
            return

        def submit_response():
            nonlocal current_index  # Deklariere zuerst nonlocal
            user_responses = [var.get() for var in vote_vars]
            self.results.loc[current_index, user_names] = user_responses
            current_index += 1
            if current_index >= len(self.comments):
                messagebox.showinfo("Fertig", "Alle Kommentare klassifiziert!")
                classify_window.destroy()
                return
            reset_button_colors()
            update_comment()

        def reset_button_colors():
            for radio_buttons in all_radio_buttons:
                for button in radio_buttons:
                    button.configure(bg="white")

        def update_comment():
            comment_label.config(text=self.comments[current_index])

        # Neues Fenster für Klassifizierung
        classify_window = tk.Toplevel(self.root)
        classify_window.title("Kommentare klassifizieren")
        classify_window.geometry("1000x800")
        classify_window.configure(bg="lightyellow")

        # Überschrift
        header_label = tk.Label(
            classify_window,
            text="Jetzt abstimmen",
            font=("Helvetica", 24, "bold"),
            fg="darkred",
            bg="lightyellow"
        )
        header_label.pack(pady=20)

        user_names = ['Max', 'Jakob', 'Kai', 'Eliano', 'Maria', 'Natalya']
        vote_vars = [tk.StringVar(value="") for _ in user_names]
        all_radio_buttons = []

        current_index = 0

        # Kommentar anzeigen
        comment_label = tk.Label(
            classify_window,
            text="",
            wraplength=900,
            justify="left",
            font=("Helvetica", 18, "italic"),
            bg="lightyellow",
            padx=10,
            pady=10,
            relief="solid"
        )
        comment_label.pack(pady=20)

        # Buttons für jede Person
        for i, name in enumerate(user_names):
            frame = tk.Frame(classify_window, bg="lightyellow")
            frame.pack(pady=10)
            tk.Label(
                frame,
                text=f"{name}",
                font=("Helvetica", 16, "bold"),
                bg="lightyellow",
                fg="darkblue"
            ).pack(side="left", padx=10)
            radio_buttons = []
            for option in ["Pro Trump", "Pro Harris", "Neutral"]:
                button = tk.Radiobutton(
                    frame,
                    text=option,
                    variable=vote_vars[i],
                    value=option,
                    font=("Helvetica", 14),
                    bg="white",
                    fg="black",
                    indicatoron=0,
                    width=15,
                    height=2,
                    relief="raised"
                )
                button.pack(side="left", padx=5)
                radio_buttons.append(button)
            all_radio_buttons.append(radio_buttons)

        submit_button = tk.Button(
            classify_window,
            text="Antwort speichern",
            font=("Helvetica", 18),
            bg="orange",
            fg="white",
            width=20,
            command=submit_response
        )
        submit_button.pack(pady=30)

        update_comment()

    def visualize_results(self):
        if self.results.isnull().any().any():
            messagebox.showerror("Fehler", "Nicht alle Kommentare wurden klassifiziert!")
            return

        summary = self.results.iloc[:, 1:].apply(pd.Series.value_counts).sum(axis=1)

        # Diagramm erstellen
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(summary.index, summary.values, color=['blue', 'orange', 'green'], edgecolor="black")
        ax.set_title("Klassifikationsübersicht", fontsize=24)
        ax.set_ylabel("Anzahl der Stimmen", fontsize=18)
        ax.set_xlabel("Kategorie", fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        # GUI-integriertes Matplotlib-Plot
        chart_window = tk.Toplevel(self.root)
        chart_window.title("Ergebnisse")
        chart_window.geometry("1000x800")
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def save_results(self):
        if self.results.empty:
            messagebox.showerror("Fehler", "Keine Ergebnisse zum Speichern vorhanden!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV-Dateien", "*.csv"), ("Alle Dateien", "*.*")]
        )
        if not file_path:
            return

        try:
            self.results.to_csv(file_path, index=False, encoding="utf-8")
            messagebox.showinfo("Erfolg", f"Ergebnisse erfolgreich unter {file_path} gespeichert!")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern der Datei: {e}")

# Hauptprogramm
if __name__ == "__main__":
    root = tk.Tk()
    app = RedditClassifierApp(root)
    root.mainloop()
