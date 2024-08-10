from visualizations import MainMenu
import tkinter as tk

if __name__ == '__main__':
    main_window = tk.Tk()
    main_window.title("Ricochet Rumble")
    MainMenu(main_window)
    main_window.mainloop()
