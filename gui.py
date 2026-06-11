from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

# O import abaixo dispara o shim de torchvision e carrega torch/basicsr/etc.
# (pesado). Por isso fazemos preguiçosamente, dentro do worker, para não
# travar a abertura da janela.
import topaz_killer_upscaler as tk_up


# ===========================================================================
# Ponte de logging -> fila da GUI
# ===========================================================================
class _QueueLogHandler(logging.Handler):
    """Encaminha registros do logger do pipeline para a fila da UI."""

    def __init__(self, q: "queue.Queue[tuple[str, object]]") -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put(("log", self.format(record)))
        except Exception:  # pragma: no cover - nunca derruba o pipeline
            pass


# ===========================================================================
# GUI
# ===========================================================================
class UpscalerGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Topaz-Killer Upscaler (Real-ESRGAN + GFPGAN)")
        self.root.minsize(640, 640)

        self.files: list[Path] = []
        self.output_dir: Path | None = None
        self.model_path: Path | None = None  # .pth customizado
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._running = False

        outer = ttk.Frame(root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        # --- Lista de imagens --------------------------------------------
        ttk.Label(outer, text="Imagens (várias ao mesmo tempo)").pack(anchor=tk.W)
        list_fr = ttk.Frame(outer)
        list_fr.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        scroll = ttk.Scrollbar(list_fr)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(
            list_fr, height=8, selectmode=tk.EXTENDED, yscrollcommand=scroll.set
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.listbox.yview)

        row_btns = ttk.Frame(outer)
        row_btns.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(row_btns, text="Adicionar arquivos…", command=self._add_files).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(row_btns, text="Adicionar pasta…", command=self._add_folder).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(
            row_btns, text="Remover selecionados", command=self._remove_selected
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row_btns, text="Limpar lista", command=self._clear_list).pack(
            side=tk.LEFT
        )

        # --- Pasta de saída ----------------------------------------------
        out_fr = ttk.Frame(outer)
        out_fr.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(out_fr, text="Pasta de saída:").pack(anchor=tk.W)
        out_row = ttk.Frame(out_fr)
        out_row.pack(fill=tk.X, pady=(2, 0))
        self.lbl_out = ttk.Label(out_row, text="(não definida)", foreground="#555")
        self.lbl_out.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_row, text="Escolher…", command=self._choose_output).pack(
            side=tk.RIGHT
        )

        # --- Opções (em duas colunas) ------------------------------------
        opts = ttk.LabelFrame(outer, text="Opções", padding=8)
        opts.pack(fill=tk.X, pady=(0, 8))
        opts.columnconfigure(1, weight=1)
        opts.columnconfigure(3, weight=1)

        r = 0
        # Modelo
        ttk.Label(opts, text="Modelo:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.var_model = tk.StringVar(value=tk_up.MODEL_NAME)
        ttk.Combobox(
            opts,
            textvariable=self.var_model,
            values=["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"],
            state="readonly",
            width=24,
        ).grid(row=r, column=1, sticky=tk.W, padx=(6, 16))
        ttk.Label(opts, text="num_block:").grid(row=r, column=2, sticky=tk.W)
        self.var_numblock = tk.StringVar(value=str(tk_up.NUM_BLOCK))
        ttk.Entry(opts, textvariable=self.var_numblock, width=6).grid(
            row=r, column=3, sticky=tk.W, padx=(6, 0)
        )

        r += 1
        # .pth customizado
        ttk.Label(opts, text="Pesos .pth (opcional):").grid(
            row=r, column=0, sticky=tk.W, pady=2
        )
        self.lbl_model_path = ttk.Label(opts, text="(usar modelo acima)", foreground="#555")
        self.lbl_model_path.grid(row=r, column=1, columnspan=2, sticky=tk.W, padx=(6, 0))
        ttk.Button(opts, text="Escolher…", command=self._choose_model).grid(
            row=r, column=3, sticky=tk.E
        )

        r += 1
        # Tile / overlap
        ttk.Label(opts, text="Tile (0=off):").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.var_tile = tk.StringVar(value=str(tk_up.TILE_SIZE))
        ttk.Entry(opts, textvariable=self.var_tile, width=6).grid(
            row=r, column=1, sticky=tk.W, padx=(6, 16)
        )
        ttk.Label(opts, text="Overlap:").grid(row=r, column=2, sticky=tk.W)
        self.var_overlap = tk.StringVar(value=str(tk_up.TILE_OVERLAP))
        ttk.Entry(opts, textvariable=self.var_overlap, width=6).grid(
            row=r, column=3, sticky=tk.W, padx=(6, 0)
        )

        r += 1
        # Rosto
        self.var_face = tk.BooleanVar(value=tk_up.ENABLE_FACE_ENHANCE)
        ttk.Checkbutton(
            opts, text="Restauração facial (GFPGAN)", variable=self.var_face
        ).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Label(opts, text="Fidelidade (0–1):").grid(row=r, column=2, sticky=tk.W)
        self.var_fidelity = tk.StringVar(value=str(tk_up.FACE_FIDELITY_WEIGHT))
        ttk.Entry(opts, textvariable=self.var_fidelity, width=6).grid(
            row=r, column=3, sticky=tk.W, padx=(6, 0)
        )

        r += 1
        # Denoise
        ttk.Label(opts, text="Denoise (0=off):").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.var_denoise = tk.StringVar(value=str(tk_up.DENOISE_STRENGTH))
        ttk.Entry(opts, textvariable=self.var_denoise, width=6).grid(
            row=r, column=1, sticky=tk.W, padx=(6, 16)
        )
        self.var_auto_denoise = tk.BooleanVar(value=tk_up.AUTO_DENOISE)
        ttk.Checkbutton(
            opts, text="Auto-denoise (ruído alto)", variable=self.var_auto_denoise
        ).grid(row=r, column=2, columnspan=2, sticky=tk.W)

        r += 1
        # Detalhe / sharpen
        ttk.Label(opts, text="Detalhe IA (0–1):").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.var_detail = tk.StringVar(value=str(tk_up.DETAIL_STRENGTH))
        ttk.Entry(opts, textvariable=self.var_detail, width=6).grid(
            row=r, column=1, sticky=tk.W, padx=(6, 16)
        )
        ttk.Label(opts, text="Sharpen (0=off):").grid(row=r, column=2, sticky=tk.W)
        self.var_sharpen = tk.StringVar(value=str(tk_up.SHARPEN_AMOUNT))
        ttk.Entry(opts, textvariable=self.var_sharpen, width=6).grid(
            row=r, column=3, sticky=tk.W, padx=(6, 0)
        )

        r += 1
        # Resolução alvo
        ttk.Label(opts, text="Largura alvo:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.var_width = tk.StringVar(
            value=str(tk_up.TARGET_WIDTH or "")
        )
        ttk.Entry(opts, textvariable=self.var_width, width=8).grid(
            row=r, column=1, sticky=tk.W, padx=(6, 16)
        )
        ttk.Label(opts, text="Altura alvo:").grid(row=r, column=2, sticky=tk.W)
        self.var_height = tk.StringVar(
            value=str(tk_up.TARGET_HEIGHT or "")
        )
        ttk.Entry(opts, textvariable=self.var_height, width=8).grid(
            row=r, column=3, sticky=tk.W, padx=(6, 0)
        )

        r += 1
        # Modo de resize / formato / fp16
        ttk.Label(opts, text="Resize:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.var_resize = tk.StringVar(value=tk_up.RESIZE_MODE)
        ttk.Combobox(
            opts,
            textvariable=self.var_resize,
            values=["fit", "exact"],
            state="readonly",
            width=8,
        ).grid(row=r, column=1, sticky=tk.W, padx=(6, 16))
        ttk.Label(opts, text="Formato:").grid(row=r, column=2, sticky=tk.W)
        self.var_format = tk.StringVar(value=tk_up.OUTPUT_FORMAT)
        ttk.Combobox(
            opts,
            textvariable=self.var_format,
            values=["png", "jpg"],
            state="readonly",
            width=8,
        ).grid(row=r, column=3, sticky=tk.W, padx=(6, 0))

        r += 1
        self.var_fp16 = tk.StringVar(value="auto")
        ttk.Label(opts, text="Precisão:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(
            opts,
            textvariable=self.var_fp16,
            values=["auto", "fp16", "fp32"],
            state="readonly",
            width=8,
        ).grid(row=r, column=1, sticky=tk.W, padx=(6, 16))
        self.var_overwrite = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opts, text="Sobrescrever existentes", variable=self.var_overwrite
        ).grid(row=r, column=2, columnspan=2, sticky=tk.W)

        # --- Ação / progresso / log --------------------------------------
        self.btn_run = ttk.Button(
            outer, text="Aumentar resolução agora", command=self._start_batch
        )
        self.btn_run.pack(fill=tk.X, pady=(0, 8))

        self.progress = ttk.Progressbar(outer, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(outer, text="Log").pack(anchor=tk.W)
        self.txt = scrolledtext.ScrolledText(
            outer, height=8, wrap=tk.WORD, state=tk.DISABLED
        )
        self.txt.pack(fill=tk.BOTH, expand=True)

        self.root.after(120, self._poll_queue)

    # ------------------------------------------------------------------ UI
    def _log(self, line: str) -> None:
        self.txt.configure(state=tk.NORMAL)
        self.txt.insert(tk.END, line + "\n")
        self.txt.see(tk.END)
        self.txt.configure(state=tk.DISABLED)

    def _refresh_listbox(self) -> None:
        self.listbox.delete(0, tk.END)
        for p in self.files:
            self.listbox.insert(tk.END, str(p))

    def _is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in tk_up.SUPPORTED_EXTENSIONS

    def _add_unique(self, path: Path) -> None:
        path = path.resolve()
        if path in self.files:
            return
        if path.is_file() and self._is_supported(path):
            self.files.append(path)

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Selecionar imagens",
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp"),
                ("Todos", "*.*"),
            ],
        )
        for p in paths:
            self._add_unique(Path(p))
        self._refresh_listbox()

    def _add_folder(self) -> None:
        d = filedialog.askdirectory(title="Pasta com imagens")
        if not d:
            return
        try:
            found = tk_up.collect_images(Path(d))
        except Exception as e:  # noqa: BLE001
            messagebox.showwarning("Pasta", str(e))
            return
        if not found:
            messagebox.showinfo("Pasta", "Nenhuma imagem suportada encontrada.")
            return
        for p in found:
            self._add_unique(p)
        self._refresh_listbox()

    def _remove_selected(self) -> None:
        sel = list(self.listbox.curselection())
        for idx in reversed(sel):
            if 0 <= idx < len(self.files):
                del self.files[idx]
        self._refresh_listbox()

    def _clear_list(self) -> None:
        self.files.clear()
        self._refresh_listbox()

    def _choose_output(self) -> None:
        d = filedialog.askdirectory(title="Pasta de saída")
        if d:
            self.output_dir = Path(d).expanduser().resolve()
            self.lbl_out.config(text=str(self.output_dir), foreground="")

    def _choose_model(self) -> None:
        f = filedialog.askopenfilename(
            title="Pesos .pth (opcional)",
            filetypes=[("PyTorch weights", "*.pth"), ("Todos", "*.*")],
        )
        if f:
            self.model_path = Path(f).expanduser().resolve()
            self.lbl_model_path.config(text=str(self.model_path), foreground="")

    # -------------------------------------------------------------- helpers
    def _parse_int(self, var: tk.StringVar, name: str, lo: int, hi: int) -> int:
        try:
            v = int(var.get().strip())
        except ValueError:
            raise ValueError(f"{name}: use um número inteiro entre {lo} e {hi}.")
        if not (lo <= v <= hi):
            raise ValueError(f"{name}: o valor deve estar entre {lo} e {hi}.")
        return v

    def _parse_float(self, var: tk.StringVar, name: str, lo: float, hi: float) -> float:
        try:
            v = float(var.get().strip())
        except ValueError:
            raise ValueError(f"{name}: use um número entre {lo} e {hi}.")
        if not (lo <= v <= hi):
            raise ValueError(f"{name}: o valor deve estar entre {lo} e {hi}.")
        return v

    def _parse_dim(self, var: tk.StringVar, name: str) -> int | None:
        raw = var.get().strip()
        if raw == "" or raw == "0":
            return None
        try:
            v = int(raw)
        except ValueError:
            raise ValueError(f"{name}: deixe vazio/0 ou use um inteiro positivo.")
        if v <= 0:
            return None
        return v

    # --------------------------------------------------------------- batch
    def _start_batch(self) -> None:
        if self._running:
            return
        if not self.files:
            messagebox.showinfo("Lista vazia", "Adicione pelo menos uma imagem.")
            return
        if not self.output_dir:
            messagebox.showinfo("Saída", "Escolha a pasta de saída.")
            return

        try:
            num_block = self._parse_int(self.var_numblock, "num_block", 1, 64)
            tile = self._parse_int(self.var_tile, "Tile", 0, 8192)
            overlap = self._parse_int(self.var_overlap, "Overlap", 0, 1024)
            fidelity = self._parse_float(self.var_fidelity, "Fidelidade", 0.0, 1.0)
            denoise = self._parse_int(self.var_denoise, "Denoise", 0, 30)
            detail = self._parse_float(self.var_detail, "Detalhe IA", 0.0, 1.0)
            sharpen = self._parse_float(self.var_sharpen, "Sharpen", 0.0, 5.0)
            target_w = self._parse_dim(self.var_width, "Largura alvo")
            target_h = self._parse_dim(self.var_height, "Altura alvo")
        except ValueError as e:
            messagebox.showerror("Opção inválida", str(e))
            return

        fp16_map = {"auto": None, "fp16": True, "fp32": False}
        cfg = {
            "model_name": self.var_model.get(),
            "model_path": str(self.model_path) if self.model_path else None,
            "num_block": num_block,
            "tile": tile,
            "overlap": overlap,
            "enable_face": bool(self.var_face.get()),
            "fp16": fp16_map[self.var_fp16.get()],
        }
        opts = tk_up.ProcessOptions(
            fidelity=fidelity,
            denoise_strength=denoise,
            auto_analyze=True,
            auto_denoise=bool(self.var_auto_denoise.get()),
            detail=detail,
            sharpen=sharpen,
            target_w=target_w,
            target_h=target_h,
            resize_mode=self.var_resize.get(),
            out_fmt=self.var_format.get(),
            overwrite=bool(self.var_overwrite.get()),
        )

        self._running = True
        self.btn_run.configure(state=tk.DISABLED)
        self.progress.configure(maximum=len(self.files), value=0)
        self.txt.configure(state=tk.NORMAL)
        self.txt.delete("1.0", tk.END)
        self.txt.configure(state=tk.DISABLED)

        files_copy = list(self.files)
        out_dir = self.output_dir
        threading.Thread(
            target=self._worker, args=(files_copy, out_dir, cfg, opts), daemon=True
        ).start()

    def _worker(
        self,
        files: list[Path],
        out_dir: Path,
        cfg: dict,
        opts: "tk_up.ProcessOptions",
    ) -> None:
        # Logger do pipeline + ponte para a UI.
        logger = tk_up.setup_logging()
        handler = _QueueLogHandler(self._queue)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        errors: list[str] = []
        try:
            device = tk_up.Device.auto(force_fp16=cfg["fp16"])
            self._queue.put(("log", f"Dispositivo: {device.describe()}"))
            self._queue.put(("log", "Carregando modelos (pode demorar na 1ª vez)…"))

            pipeline = tk_up.UpscalePipeline.build(
                device=device,
                logger=logger,
                model_path=cfg["model_path"],
                model_name=cfg["model_name"],
                netscale=tk_up.NETSCALE,
                num_block=cfg["num_block"],
                tile=cfg["tile"],
                overlap=cfg["overlap"],
                enable_face=cfg["enable_face"],
            )
            base_tile = pipeline.upsampler.tile

            out_dir.mkdir(parents=True, exist_ok=True)
            self._queue.put(("log", f"Processando {len(files)} imagem(ns)…"))

            done = 0
            for src in files:
                out_path = out_dir / f"{src.stem}.{opts.out_fmt}"
                if out_path.exists() and not opts.overwrite:
                    done += 1
                    self._queue.put(("log", f"SKIP (já existe): {src.name}"))
                    self._queue.put(("prog", done))
                    continue
                t0 = time.perf_counter()
                try:
                    info = tk_up.process_one(pipeline, src, out_path, opts, logger)
                    dt = time.perf_counter() - t0
                    self._queue.put(("log", f"OK: {src.name} | {info} ({dt:.1f}s)"))
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{src.name}: {e}")
                    self._queue.put(("log", f"ERRO: {src.name}: {e}"))
                    logger.debug(traceback.format_exc())
                finally:
                    pipeline.set_tile(base_tile)
                    pipeline.device.empty_cache()
                    done += 1
                    self._queue.put(("prog", done))
        except Exception as e:  # noqa: BLE001 — falha ao montar pipeline
            errors.append(f"pipeline: {e}")
            self._queue.put(("log", f"ERRO ao iniciar: {e}"))
            logger.debug(traceback.format_exc())
        finally:
            logger.removeHandler(handler)
            self._queue.put(("done", errors))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self._log(str(payload))
                elif kind == "prog":
                    self.progress["value"] = int(payload)  # type: ignore[arg-type]
                elif kind == "done":
                    errs = payload  # type: ignore[assignment]
                    self._running = False
                    self.btn_run.configure(state=tk.NORMAL)
                    if errs:
                        messagebox.showwarning(
                            "Concluído com erros",
                            f"{len(errs)} falha(s). Veja o log para detalhes.",
                        )
                    else:
                        messagebox.showinfo(
                            "Concluído", "Todas as imagens foram processadas."
                        )
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)


def main() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except tk.TclError:
        pass
    UpscalerGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
