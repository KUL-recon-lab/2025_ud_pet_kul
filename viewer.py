import numpy as np
import array_api_compat.torch as xp
import matplotlib.pyplot as plt
import matplotlib.widgets as mw

from matplotlib._pylab_helpers import Gcf

# TODO:
# - export movie looping over i,j,k,t,g


class ThreeAxisViewer:
    def __init__(
        self,
        volume,
        affine=np.eye(4),
        base_fig_width=4.0,
        show_axis_labels=False,
        add_colorbar=True,
        cbar_kws=dict(location="right", fraction=0.03),
        **kwargs,
    ):
        plt.ion()

        if volume.ndim == 5:
            self._volume = volume
        elif volume.ndim == 4:
            self._volume = xp.expand_dims(volume, axis=0)
        elif volume.ndim == 3:
            self._volume = xp.expand_dims(xp.expand_dims(volume, axis=0), axis=0)
        else:
            raise ValueError("Input volume must be 3D, 4D or 5D.")

        self._affine = affine

        self._voxel_size_i = float(np.sqrt((affine[:3, 0] ** 2).sum()))
        self._voxel_size_j = float(np.sqrt((affine[:3, 1] ** 2).sum()))
        self._voxel_size_k = float(np.sqrt((affine[:3, 2] ** 2).sum()))

        self._kwargs = kwargs

        self._ng, self._nt, self._n_i, self._n_j, self._n_k = self._volume.shape

        self._ig = 0
        self._it = 0
        self._i = self._n_i // 2
        self._j = self._n_j // 2
        self._k = self._n_k // 2

        self._aspect_ratio_i = self._voxel_size_k / self._voxel_size_j
        self._aspect_ratio_j = self._voxel_size_k / self._voxel_size_i
        self._aspect_ratio_k = self._voxel_size_j / self._voxel_size_i

        self._base_fig_width = base_fig_width

        self._fig_k_width = self._base_fig_width
        self._fig_k_height = (
            self._fig_k_width
            * (self._n_j * self._voxel_size_j)
            / (self._n_i * self._voxel_size_i)
        )

        self._fig_j_width = self._fig_k_width
        self._fig_j_height = (
            self._fig_j_width
            * (self._n_k * self._voxel_size_k)
            / (self._n_i * self._voxel_size_i)
        )

        self._fig_i_height = self._fig_j_height
        self._fig_i_width = (
            self._fig_i_height
            * (self._n_j * self._voxel_size_j)
            / (self._n_k * self._voxel_size_k)
        )

        self._fig_i, self._ax_i = plt.subplots(
            1, 1, figsize=(self._fig_i_width, self._fig_i_height), layout="constrained"
        )
        self._fig_j, self._ax_j = plt.subplots(
            1, 1, figsize=(self._fig_j_width, self._fig_j_height), layout="constrained"
        )
        self._fig_k, self._ax_k = plt.subplots(
            1, 1, figsize=(self._fig_k_width, self._fig_k_height), layout="constrained"
        )

        self._fig_mcor, self._ax_mcor = plt.subplots(
            1, 1, figsize=(self._fig_j_width, self._fig_j_height), layout="constrained"
        )

        self._fig_msag, self._ax_msag = plt.subplots(
            1, 1, figsize=(self._fig_i_width, self._fig_i_height), layout="constrained"
        )

        self._img_i = self._ax_i.imshow(
            self._volume[self._ig, self._it, self._i, :, :].T,
            origin="lower",
            aspect=self._aspect_ratio_i,
            **self._kwargs,
        )
        self._img_j = self._ax_j.imshow(
            self._volume[self._ig, self._it, :, self._j, :].T,
            origin="lower",
            **self._kwargs,
            aspect=self._aspect_ratio_j,
        )
        self._img_k = self._ax_k.imshow(
            self._volume[self._ig, self._it, :, :, self._k].T,
            origin="lower",
            **self._kwargs,
            aspect=self._aspect_ratio_k,
        )

        self._img_msag = self._ax_msag.imshow(
            xp.max(self._volume[self._ig, self._it, :, :, :], axis=0).T,
            origin="lower",
            aspect=self._aspect_ratio_i,
            **self._kwargs,
        )

        self._img_mcor = self._ax_mcor.imshow(
            xp.max(self._volume[self._ig, self._it, :, :, :], axis=1).T,
            origin="lower",
            aspect=self._aspect_ratio_i,
            **self._kwargs,
        )

        self._ax_i.set_title(
            f"[g={self._ig},t={self._it},i={self._i},j,k]", fontsize="small"
        )
        self._ax_j.set_title(
            f"[g={self._ig},t={self._it},i,j={self._j},k]", fontsize="small"
        )
        self._ax_k.set_title(
            f"[g={self._ig},t={self._it},i,j,k={self._k}]", fontsize="small"
        )

        self._ax_msag.set_title(
            f"[g={self._ig},t={self._it}, ...] mip", fontsize="small"
        )
        self._ax_mcor.set_title(
            f"[g={self._ig},t={self._it}, ...] mip", fontsize="small"
        )

        if show_axis_labels:
            self._ax_i.set_xlabel("j", fontsize="small")
            self._ax_i.set_ylabel("k", fontsize="small")

            self._ax_j.set_xlabel("i", fontsize="small")
            self._ax_j.set_ylabel("k", fontsize="small")

            self._ax_k.set_xlabel("i", fontsize="small")
            self._ax_k.set_ylabel("j", fontsize="small")
        else:
            for axx in [self._ax_i, self._ax_j, self._ax_k]:
                axx.tick_params(axis="x", labelsize="small")
                axx.tick_params(axis="y", labelsize="small")

            for axx in [
                self._ax_i,
                self._ax_j,
                self._ax_k,
                self._ax_mcor,
                self._ax_msag,
            ]:
                axx.set_xticks([])
                axx.set_yticks([])

        if add_colorbar:
            cbar_i = self._fig_i.colorbar(self._img_i, ax=self._ax_i, **cbar_kws)
            cbar_j = self._fig_j.colorbar(self._img_j, ax=self._ax_j, **cbar_kws)
            cbar_k = self._fig_k.colorbar(self._img_k, ax=self._ax_k, **cbar_kws)
            cbar_mcor = self._fig_mcor.colorbar(
                self._img_mcor, ax=self._ax_mcor, **cbar_kws
            )
            cbar_msag = self._fig_msag.colorbar(
                self._img_msag, ax=self._ax_msag, **cbar_kws
            )
            cbar_i.ax.tick_params(labelsize="small")
            cbar_j.ax.tick_params(labelsize="small")
            cbar_k.ax.tick_params(labelsize="small")
            cbar_mcor.ax.tick_params(labelsize="small")
            cbar_msag.ax.tick_params(labelsize="small")

        self._fig_i.show()
        self._fig_j.show()
        self._fig_k.show()
        self._fig_mcor.show()
        self._fig_msag.show()

        # add interactivity
        self.connect_scroll_events()
        self.connect_key_events()
        # allow changing slices by dragging the mouse
        self.connect_drag_events()

        # --- control panel: sliders for vmin/vmax and cmap selector ---
        # default display limits
        self._vmin = 0.0
        self._vmax = 20.0
        self._cmap = "Greys"

        try:
            # three stacked axes: vmin slider, vmax slider, cmap selector
            self._fig_ctrl, (
                self._ax_ctrl_vmin,
                self._ax_ctrl_vmax,
                self._ax_ctrl_cmap,
            ) = plt.subplots(
                3,
                1,
                figsize=(3, 4),
                layout="constrained",
            )
        except Exception:
            # fallback: create a single figure and add axes manually
            self._fig_ctrl = plt.figure(figsize=(3, 4), constrained_layout=True)
            self._ax_ctrl_vmin = self._fig_ctrl.add_axes((0.15, 0.67, 0.7, 0.2))
            self._ax_ctrl_vmax = self._fig_ctrl.add_axes((0.15, 0.37, 0.7, 0.2))
            self._ax_ctrl_cmap = self._fig_ctrl.add_axes((0.15, 0.07, 0.7, 0.2))

        # sliders
        self._slider_vmin = mw.Slider(
            self._ax_ctrl_vmin, "vmin", 0.0, 20.0, valinit=self._vmin
        )
        self._slider_vmax = mw.Slider(
            self._ax_ctrl_vmax, "vmax", 0.0, 20.0, valinit=self._vmax
        )

        # cmap selector: prefer Dropdown if available, else fallback to RadioButtons
        try:
            Dropdown = getattr(mw, "Dropdown")
        except Exception:
            Dropdown = None

        if Dropdown is not None:
            # use full list of available colormaps
            cmaps = list(plt.colormaps())
            try:
                self._dropdown_cmap = Dropdown(
                    self._ax_ctrl_cmap, "cmap", cmaps, value=self._cmap
                )
                self._use_dropdown = True
            except Exception:
                self._use_dropdown = False
                # fallback
                self._radio_cmap = mw.RadioButtons(
                    self._ax_ctrl_cmap, ["Greys", "viridis", "plasma", "magma"]
                )
        else:
            self._use_dropdown = False
            self._radio_cmap = mw.RadioButtons(
                self._ax_ctrl_cmap, ["Greys", "viridis", "plasma", "magma"]
            )

        # apply settings to all images
        def _apply_clim_and_cmap():
            imgs = [
                getattr(self, "_img_i", None),
                getattr(self, "_img_j", None),
                getattr(self, "_img_k", None),
                getattr(self, "_img_mcor", None),
                getattr(self, "_img_msag", None),
            ]
            for im in imgs:
                if im is None:
                    continue
                try:
                    im.set_clim(self._vmin, self._vmax)
                except Exception:
                    pass
                try:
                    im.set_cmap(self._cmap)
                except Exception:
                    pass
            try:
                self.redraw_all()
            except Exception:
                pass

        # slider callbacks
        def _on_vmin(val):
            try:
                self._vmin = float(val)
            except Exception:
                return
            # ensure vmin <= vmax
            if self._vmin > self._vmax:
                # push vmax up to vmin
                try:
                    self._slider_vmax.set_val(self._vmin)
                except Exception:
                    self._vmax = self._vmin
            _apply_clim_and_cmap()

        def _on_vmax(val):
            try:
                self._vmax = float(val)
            except Exception:
                return
            if self._vmax < self._vmin:
                try:
                    self._slider_vmin.set_val(self._vmax)
                except Exception:
                    self._vmin = self._vmax
            _apply_clim_and_cmap()

        self._slider_vmin.on_changed(_on_vmin)
        self._slider_vmax.on_changed(_on_vmax)

        # cmap callbacks
        if getattr(self, "_use_dropdown", False):
            try:

                def _on_dropdown(label):
                    # Dropdown.value is the selection in newer mpl; label may be unused
                    try:
                        sel = self._dropdown_cmap.value
                    except Exception:
                        sel = label
                    self._cmap = sel
                    _apply_clim_and_cmap()

                self._dropdown_cmap.on_changed(_on_dropdown)
            except Exception:
                # fallback to radio
                self._radio_cmap.on_clicked(
                    lambda label: (
                        setattr(self, "_cmap", label),
                        _apply_clim_and_cmap(),
                    )
                )
        else:
            self._radio_cmap.on_clicked(
                lambda label: (setattr(self, "_cmap", label), _apply_clim_and_cmap())
            )

        # show control panel
        try:
            self._fig_ctrl.show()
        except Exception:
            pass

        # for fig_i, fig_j, fig_k show a single x and y tick corresponding to the current i, j, k
        self._ax_i.set_xticks([self._j])
        self._ax_i.set_yticks([self._k])
        self._ax_i.set_xticklabels([""])
        self._ax_i.set_yticklabels([""])

        self._ax_j.set_xticks([self._i])
        self._ax_j.set_yticks([self._k])
        self._ax_j.set_xticklabels([""])
        self._ax_j.set_yticklabels([""])

        self._ax_k.set_xticks([self._i])
        self._ax_k.set_yticks([self._j])
        self._ax_k.set_xticklabels([""])
        self._ax_k.set_yticklabels([""])

    # destructor
    def __del__(self):
        self.close()

    @property
    def volume(self):
        return self._volume

    @property
    def affine(self):
        return self._affine

    @property
    def voxel_size_i(self):
        return self._voxel_size_i

    @property
    def voxel_size_j(self):
        return self._voxel_size_j

    @property
    def voxel_size_k(self):
        return self._voxel_size_k

    @property
    def base_fig_width(self):
        return self._base_fig_width

    @property
    def fig_i(self):
        return self._fig_i

    @property
    def fig_j(self):
        return self._fig_j

    @property
    def fig_k(self):
        return self._fig_k

    @property
    def ax_i(self):
        return self._ax_i

    @property
    def ax_j(self):
        return self._ax_j

    @property
    def ax_k(self):
        return self._ax_k

    @property
    def nt(self):
        return self._nt

    @property
    def ng(self):
        return self._ng

    @property
    def n_i(self):
        return self._n_i

    @property
    def n_j(self):
        return self._n_j

    @property
    def n_k(self):
        return self._n_k

    @property
    def i(self) -> int:
        return self._i

    @i.setter
    def i(self, value: int) -> None:
        self._i = value
        self.redraw_img_i()

    @property
    def j(self) -> int:
        return self._j

    @j.setter
    def j(self, value: int) -> None:
        self._j = value
        self.redraw_img_j()

    @property
    def k(self) -> int:
        return self._k

    @k.setter
    def k(self, value: int) -> None:
        self._k = value
        self.redraw_img_k()

    @property
    def it(self) -> int:
        return self._it

    @it.setter
    def it(self, value: int) -> None:
        self._it = value
        self.redraw_all()

    @property
    def ig(self) -> int:
        return self._ig

    @ig.setter
    def ig(self, value: int) -> None:
        self._ig = value
        self.redraw_all()

    @property
    def img_i(self):
        return self._img_i

    @property
    def img_j(self):
        return self._img_j

    @property
    def img_k(self):
        return self._img_k

    def redraw_img_i(self):
        if self._fig_k in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.figure(self._fig_k.number)
            self._ax_k.set_xticks([self._i])
            plt.draw()
        if self._fig_j in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.figure(self._fig_j.number)
            self._ax_j.set_xticks([self._i])
            plt.draw()

        if self._fig_i in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.figure(self._fig_i.number)
            self._img_i.set_data(self._volume[self._ig, self._it, self._i, :, :].T)
            self._ax_i.set_title(
                f"[g={self._ig},t={self._it},i={self._i},j,k]", fontsize="small"
            )
            plt.draw()

    def redraw_img_j(self):
        if self._fig_i in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.figure(self._fig_i.number)
            self._ax_i.set_xticks([self._j])
            plt.draw()
        if self._fig_k in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.figure(self._fig_k.number)
            self._ax_k.set_yticks([self._j])
            plt.draw()
        if self._fig_j in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.figure(self._fig_j.number)
            self._img_j.set_data(self._volume[self._ig, self._it, :, self._j, :].T)
            self._ax_j.set_title(
                f"[g={self._ig},t={self._it},i,j={self._j},k]", fontsize="small"
            )
            plt.draw()

    def redraw_img_k(self):
        if self._fig_i in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.figure(self._fig_i.number)
            self._ax_i.set_yticks([self._k])
            plt.draw()
        if self._fig_j in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.figure(self._fig_j.number)
            self._ax_j.set_yticks([self._k])
            plt.draw()
        if self._fig_k in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.figure(self._fig_k.number)
            self._img_k.set_data(self._volume[self._ig, self._it, :, :, self._k].T)
            self._ax_k.set_title(
                f"[g={self._ig},t={self._it},i,j,k={self._k}]", fontsize="small"
            )
            plt.draw()

    def redraw_all(self):
        self.redraw_img_i()
        self.redraw_img_j()
        self.redraw_img_k()

    def on_scroll(self, event):
        if event.inaxes == self._ax_i:
            if event.button == "up":
                self.i = (self._i + 1) % self._n_i
            elif event.button == "down":
                self.i = (self._i - 1) % self._n_i
        elif event.inaxes == self._ax_j:
            if event.button == "up":
                self.j = (self._j + 1) % self._n_j
            elif event.button == "down":
                self.j = (self._j - 1) % self._n_j
        elif event.inaxes == self._ax_k:
            if event.button == "up":
                self.k = (self._k + 1) % self._n_k
            elif event.button == "down":
                self.k = (self._k - 1) % self._n_k

    def connect_scroll_events(self):
        self._cid_scroll_i = self._fig_i.canvas.mpl_connect(
            "scroll_event", self.on_scroll
        )
        self._cid_scroll_j = self._fig_j.canvas.mpl_connect(
            "scroll_event", self.on_scroll
        )
        self._cid_scroll_k = self._fig_k.canvas.mpl_connect(
            "scroll_event", self.on_scroll
        )

    def on_button_press(self, event):
        # determine which axis was clicked
        if event.inaxes == self._ax_i:
            axis = "i"
        elif event.inaxes == self._ax_j:
            axis = "j"
        elif event.inaxes == self._ax_k:
            axis = "k"
        else:
            axis = None

        # LEFT CLICK (button 1): jump orthogonal views to clicked coords
        if axis is not None and getattr(event, "button", None) == 1:
            x = event.xdata
            y = event.ydata
            if x is None or y is None:
                return
            if axis == "i":
                # ax_i: x -> j, y -> k (no flips applied)
                j = int(round(x))
                k = int(round(y))
                j = max(0, min(self._n_j - 1, j))
                k = max(0, min(self._n_k - 1, k))
                self.j = j
                self.k = k
            elif axis == "j":
                # ax_j: x -> i, y -> k
                i = int(round(x))
                k = int(round(y))
                i = max(0, min(self._n_i - 1, i))
                k = max(0, min(self._n_k - 1, k))
                self.i = i
                self.k = k
            elif axis == "k":
                # ax_k: x -> j, y -> i
                j = int(round(x))
                i = int(round(y))
                j = max(0, min(self._n_j - 1, j))
                i = max(0, min(self._n_i - 1, i))
                self.j = j
                self.i = i
            return

        # LEFT CLICK in optional multi-planar axes
        # ax_msag (or f_ax_msag): shows j horizontally and k vertically -> map x->j, y->k
        if event.inaxes == self._ax_msag and getattr(event, "button", None) == 1:
            x = event.xdata
            y = event.ydata
            if x is None or y is None:
                return
            j = int(round(x))
            k = int(round(y))
            j = max(0, min(self._n_j - 1, j))
            k = max(0, min(self._n_k - 1, k))
            self.j = j
            self.k = k
            return

        # ax_mcor (or f_ax_mcor): shows i horizontally and k vertically -> map x->i, y->k
        if event.inaxes == self._ax_mcor and getattr(event, "button", None) == 1:
            x = event.xdata
            y = event.ydata
            if x is None or y is None:
                return
            i = int(round(x))
            k = int(round(y))
            i = max(0, min(self._n_i - 1, i))
            k = max(0, min(self._n_k - 1, k))
            self.i = i
            self.k = k
            return

        # RIGHT CLICK (button 3): start dragging to change slices
        if axis is not None and getattr(event, "button", None) == 3:
            self._drag_axis = axis
        else:
            self._drag_axis = None

        if self._drag_axis is not None:
            # store start position and start index
            self._drag_start_y = event.y
            self._drag_start_x = event.x
            if self._drag_axis == "i":
                self._drag_start_index = int(self._i)
            elif self._drag_axis == "j":
                self._drag_start_index = int(self._j)
            elif self._drag_axis == "k":
                self._drag_start_index = int(self._k)

    def on_motion(self, event):
        # only process if currently dragging and we have start state
        if not hasattr(self, "_drag_axis") or self._drag_axis is None:
            return
        if not hasattr(self, "_drag_start_y") or self._drag_start_y is None:
            return
        if not hasattr(self, "_drag_start_index") or self._drag_start_index is None:
            return
        # event may be None or outside axes; just use y coordinate
        try:
            dy = event.y - self._drag_start_y
        except Exception:
            return

        # sensitivity: number of pixels per slice change
        pixels_per_slice = 6.0
        delta = int(round(-dy / pixels_per_slice))

        if self._drag_axis == "i":
            new_i = (self._drag_start_index + delta) % self._n_i
            if new_i != self._i:
                self.i = new_i
        elif self._drag_axis == "j":
            new_j = (self._drag_start_index + delta) % self._n_j
            if new_j != self._j:
                self.j = new_j
        elif self._drag_axis == "k":
            new_k = (self._drag_start_index + delta) % self._n_k
            if new_k != self._k:
                self.k = new_k

    def on_button_release(self, event):
        self._drag_axis = None
        self._drag_start_y = None
        self._drag_start_x = None
        self._drag_start_index = None

    def connect_drag_events(self):
        # connect press/motion/release for each figure
        self._cid_press_i = self._fig_i.canvas.mpl_connect(
            "button_press_event", self.on_button_press
        )
        self._cid_motion_i = self._fig_i.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )
        self._cid_release_i = self._fig_i.canvas.mpl_connect(
            "button_release_event", self.on_button_release
        )

        self._cid_press_j = self._fig_j.canvas.mpl_connect(
            "button_press_event", self.on_button_press
        )
        self._cid_motion_j = self._fig_j.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )
        self._cid_release_j = self._fig_j.canvas.mpl_connect(
            "button_release_event", self.on_button_release
        )

        self._cid_press_k = self._fig_k.canvas.mpl_connect(
            "button_press_event", self.on_button_press
        )
        self._cid_motion_k = self._fig_k.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )
        self._cid_release_k = self._fig_k.canvas.mpl_connect(
            "button_release_event", self.on_button_release
        )
        # connect key events for mcor and msag too
        self._cid_release_mcor = self._fig_mcor.canvas.mpl_connect(
            "button_press_event", self.on_button_press
        )
        self._cid_release_msag = self._fig_msag.canvas.mpl_connect(
            "button_press_event", self.on_button_press
        )

    def disconnect_drag_events(self):
        if hasattr(self, "_cid_press_i") and hasattr(self, "_fig_i"):
            self._fig_i.canvas.mpl_disconnect(self._cid_press_i)
        if hasattr(self, "_cid_motion_i") and hasattr(self, "_fig_i"):
            self._fig_i.canvas.mpl_disconnect(self._cid_motion_i)
        if hasattr(self, "_cid_release_i") and hasattr(self, "_fig_i"):
            self._fig_i.canvas.mpl_disconnect(self._cid_release_i)

        if hasattr(self, "_cid_press_j") and hasattr(self, "_fig_j"):
            self._fig_j.canvas.mpl_disconnect(self._cid_press_j)
        if hasattr(self, "_cid_motion_j") and hasattr(self, "_fig_j"):
            self._fig_j.canvas.mpl_disconnect(self._cid_motion_j)
        if hasattr(self, "_cid_release_j") and hasattr(self, "_fig_j"):
            self._fig_j.canvas.mpl_disconnect(self._cid_release_j)

        if hasattr(self, "_cid_press_k") and hasattr(self, "_fig_k"):
            self._fig_k.canvas.mpl_disconnect(self._cid_press_k)
        if hasattr(self, "_cid_motion_k") and hasattr(self, "_fig_k"):
            self._fig_k.canvas.mpl_disconnect(self._cid_motion_k)
        if hasattr(self, "_cid_release_k") and hasattr(self, "_fig_k"):
            self._fig_k.canvas.mpl_disconnect(self._cid_release_k)
        # Note: we don't connect drag (motion) events for mcor/msag, so there
        # is no need to attempt to disconnect related cids here.

    def disconnect_scroll_events(self):
        if hasattr(self, "_cid_scroll_i") and hasattr(self, "_fig_i"):
            self._fig_i.canvas.mpl_disconnect(self._cid_scroll_i)
        if hasattr(self, "_cid_scroll_j") and hasattr(self, "_fig_j"):
            self._fig_j.canvas.mpl_disconnect(self._cid_scroll_j)
        if hasattr(self, "_cid_scroll_k") and hasattr(self, "_fig_k"):
            self._fig_k.canvas.mpl_disconnect(self._cid_scroll_k)

    # --- end single-viewer drag support ---

    def on_key_press(self, event):
        if event.key == "left":
            self.it = (self._it - 1) % self._nt
        elif event.key == "right":
            self.it = (self._it + 1) % self._nt
        elif event.key == "down":
            self.ig = (self._ig - 1) % self._ng
        elif event.key == "up":
            self.ig = (self._ig + 1) % self._ng

    def connect_key_events(self):
        self._cif_key_i = self._fig_i.canvas.mpl_connect(
            "key_press_event", self.on_key_press
        )
        self._cif_key_j = self._fig_j.canvas.mpl_connect(
            "key_press_event", self.on_key_press
        )
        self._cif_key_k = self._fig_k.canvas.mpl_connect(
            "key_press_event", self.on_key_press
        )

    def disconnect_key_events(self):
        if hasattr(self, "_cif_key_i") and hasattr(self, "_fig_i"):
            self._fig_i.canvas.mpl_disconnect(self._cif_key_i)
        if hasattr(self, "_cif_key_j") and hasattr(self, "_fig_j"):
            self._fig_j.canvas.mpl_disconnect(self._cif_key_j)
        if hasattr(self, "_cif_key_k") and hasattr(self, "_fig_k"):
            self._fig_k.canvas.mpl_disconnect(self._cif_key_k)

    def close(self):
        self.disconnect_scroll_events()
        self.disconnect_key_events()

        if self._fig_i in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.close(self._fig_i)
        if self._fig_j in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.close(self._fig_j)
        if self._fig_k in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.close(self._fig_k)

        if self._fig_mcor in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.close(self._fig_mcor)

        if self._fig_msag in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            plt.close(self._fig_msag)

    def savefigs(self, base: str, **kwargs):
        if self._fig_i in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            self._fig_i.savefig(base + f"_i{self.i}.png", **kwargs)

        if self._fig_j in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            self._fig_j.savefig(base + f"_j{self.j}.png", **kwargs)

        if self._fig_k in [
            manager.canvas.figure for manager in Gcf.get_all_fig_managers()
        ]:
            self._fig_k.savefig(base + f"_k{self.k}.png", **kwargs)

    def distribute_figures(
        self, margin: int = 0, cols: int | None = None, include_mip: bool = True
    ):
        """Arrange this viewer's figures on screen in a tiled layout.

        Parameters
        - margin: pixels of margin between tiles and screen edges
        - cols: number of columns; if None, a square-like layout is chosen
        - include_mip: whether to include the mcor/msag figures in the tiling

        This tries to support common GUI backends (Qt, Tk). If the backend
        doesn't expose window geometry controls, the call is a no-op.
        """
        figs = [self._fig_i, self._fig_j, self._fig_k]
        if include_mip:
            # some viewers might not have mcor/msag (be defensive)
            if hasattr(self, "_fig_mcor") and self._fig_mcor is not None:
                figs.append(self._fig_mcor)
            if hasattr(self, "_fig_msag") and self._fig_msag is not None:
                figs.append(self._fig_msag)

        figs = [
            f
            for f in figs
            if f in [m.canvas.figure for m in Gcf.get_all_fig_managers()]
        ]
        # include control panel figure at the end of the row if present
        if hasattr(self, "_fig_ctrl") and self._fig_ctrl is not None:
            if self._fig_ctrl in [m.canvas.figure for m in Gcf.get_all_fig_managers()]:
                figs.append(self._fig_ctrl)
        n = len(figs)
        if n == 0:
            return

        # Force a single-row layout: distribute horizontally only
        cols = n
        rows = 1

        # Determine screen size from first figure's manager if possible
        sw = None
        sh = None
        mgr = figs[0].canvas.manager
        win = getattr(mgr, "window", None)
        if win is not None:
            # Qt backend
            try:
                scr = win.screen()
                geom = scr.availableGeometry()
                sw, sh = int(geom.width()), int(geom.height())
            except Exception:
                # Tk backend
                try:
                    sw = int(win.winfo_screenwidth())
                    sh = int(win.winfo_screenheight())
                except Exception:
                    sw, sh = None, None

        # fallback to a reasonable default if we couldn't get screen size
        if sw is None or sh is None:
            try:
                import ctypes

                user32 = ctypes.windll.user32
                sw, sh = int(user32.GetSystemMetrics(0)), int(
                    user32.GetSystemMetrics(1)
                )
            except Exception:
                sw, sh = 1920, 1080

        # Position figures in one row without resizing: use current canvas size
        x = margin
        for idx, fig in enumerate(figs):
            mgr = fig.canvas.manager
            win = getattr(mgr, "window", None)
            if win is None:
                # skip if we can't access the window
                continue

            # try to get figure canvas size (pixels)
            try:
                w_px, h_px = fig.canvas.get_width_height()
            except Exception:
                w_px, h_px = None, None

            # fallback to window/widget size getters
            if w_px is None or h_px is None or w_px == 0 or h_px == 0:
                try:
                    # Qt style
                    w_px = int(win.width())
                    h_px = int(win.height())
                except Exception:
                    try:
                        w_px = int(win.winfo_width())
                        h_px = int(win.winfo_height())
                    except Exception:
                        # last resort: use a default canvas size
                        w_px, h_px = 800, 600

            # align to top: place window at the margin from the top edge
            # (margin==0 will place the window at the very top)
            y = int(margin)

            # Try setting window position while preserving size
            try:
                win.setGeometry(int(x), int(y), int(w_px), int(h_px))
            except Exception:
                try:
                    win.wm_geometry(f"{int(w_px)}x{int(h_px)}+{int(x)}+{int(y)}")
                except Exception:
                    try:
                        win.geometry(f"{int(w_px)}x{int(h_px)}+{int(x)}+{int(y)}")
                    except Exception:
                        # can't position this window; skip it
                        pass

            # move to next column
            x += w_px + margin

        # small pause to allow window managers to catch up
        try:
            plt.pause(0.01)
        except Exception:
            pass


class ThreeAxisViewerLinker:
    def __init__(self, viewers: list[ThreeAxisViewer]):
        # check if the arrays x.volume for x in viewers have all the same shape
        first_shape = viewers[0].volume.shape
        if not all([first_shape == x.volume.shape for x in viewers[1:]]):
            raise ValueError("All input volumes must have the same shape.")

        self._viewers = viewers

        for viewer in self._viewers:
            viewer.disconnect_scroll_events()
            viewer.disconnect_key_events()

        # disable individual viewer drag events (we'll handle dragging centrally)
        for viewer in self._viewers:
            try:
                viewer.disconnect_drag_events()
            except Exception:
                pass

        self.connect_scroll_events()
        self.connect_key_events()

    def __del__(self):
        for viewer in self._viewers:
            viewer.connect_scroll_events()
            viewer.connect_key_events()

        # enable linked drag handling
        self.connect_drag_events()

    def on_scroll(self, event):
        if any([event.inaxes == x.ax_i for x in self._viewers]):
            for viewer in self._viewers:
                if event.button == "up":
                    viewer.i = (viewer.i + 1) % viewer.n_i
                elif event.button == "down":
                    viewer.i = (viewer.i - 1) % viewer.n_i

        if any([event.inaxes == x.ax_j for x in self._viewers]):
            for viewer in self._viewers:
                if event.button == "up":
                    viewer.j = (viewer.j + 1) % viewer.n_j
                elif event.button == "down":
                    viewer.j = (viewer.j - 1) % viewer.n_j

        if any([event.inaxes == x.ax_k for x in self._viewers]):
            for viewer in self._viewers:
                if event.button == "up":
                    viewer.k = (viewer.k + 1) % viewer.n_k
                elif event.button == "down":
                    viewer.k = (viewer.k - 1) % viewer.n_k

    def connect_scroll_events(self):
        for viewer in self._viewers:
            viewer.fig_i.canvas.mpl_connect("scroll_event", self.on_scroll)
            viewer.fig_j.canvas.mpl_connect("scroll_event", self.on_scroll)
            viewer.fig_k.canvas.mpl_connect("scroll_event", self.on_scroll)

    # Drag handling for linked viewers
    def on_button_press(self, event):
        # determine which axis the press occurred in (if any)
        self._linked_drag_axis = None
        for v in self._viewers:
            if event.inaxes == v.ax_i:
                self._linked_drag_axis = "i"
                break
            if event.inaxes == v.ax_j:
                self._linked_drag_axis = "j"
                break
            if event.inaxes == v.ax_k:
                self._linked_drag_axis = "k"
                break

        if self._linked_drag_axis is not None:
            self._linked_drag_start_y = event.y
            # store start indices for all viewers
            self._linked_drag_start_indices = [(v.i, v.j, v.k) for v in self._viewers]

    def on_motion(self, event):
        if not hasattr(self, "_linked_drag_axis") or self._linked_drag_axis is None:
            return
        if (
            not hasattr(self, "_linked_drag_start_y")
            or self._linked_drag_start_y is None
        ):
            return
        if (
            not hasattr(self, "_linked_drag_start_indices")
            or self._linked_drag_start_indices is None
        ):
            return

        try:
            dy = event.y - self._linked_drag_start_y
        except Exception:
            return

        pixels_per_slice = 6.0
        delta = int(round(-dy / pixels_per_slice))

        # apply delta to the appropriate index for all viewers
        if self._linked_drag_axis == "i":
            for v, (si, sj, sk) in zip(self._viewers, self._linked_drag_start_indices):
                new_i = (si + delta) % v.n_i
                if new_i != v.i:
                    v.i = new_i
        elif self._linked_drag_axis == "j":
            for v, (si, sj, sk) in zip(self._viewers, self._linked_drag_start_indices):
                new_j = (sj + delta) % v.n_j
                if new_j != v.j:
                    v.j = new_j
        elif self._linked_drag_axis == "k":
            for v, (si, sj, sk) in zip(self._viewers, self._linked_drag_start_indices):
                new_k = (sk + delta) % v.n_k
                if new_k != v.k:
                    v.k = new_k

    def on_button_release(self, event):
        self._linked_drag_axis = None
        self._linked_drag_start_y = None
        self._linked_drag_start_indices = None

    def connect_drag_events(self):
        for viewer in self._viewers:
            viewer.fig_i.canvas.mpl_connect("button_press_event", self.on_button_press)
            viewer.fig_i.canvas.mpl_connect("motion_notify_event", self.on_motion)
            viewer.fig_i.canvas.mpl_connect(
                "button_release_event", self.on_button_release
            )

            viewer.fig_j.canvas.mpl_connect("button_press_event", self.on_button_press)
            viewer.fig_j.canvas.mpl_connect("motion_notify_event", self.on_motion)
            viewer.fig_j.canvas.mpl_connect(
                "button_release_event", self.on_button_release
            )

            viewer.fig_k.canvas.mpl_connect("button_press_event", self.on_button_press)
            viewer.fig_k.canvas.mpl_connect("motion_notify_event", self.on_motion)
            viewer.fig_k.canvas.mpl_connect(
                "button_release_event", self.on_button_release
            )

    def disconnect_drag_events(self):
        # We can't easily track connection ids here because they were not stored;
        # instead, attempt to call each viewer's disconnect_drag_events so their
        # callbacks are removed if they stored ids.
        for viewer in self._viewers:
            try:
                viewer.disconnect_drag_events()
            except Exception:
                pass

    def on_key_press(self, event):
        if event.key == "left":
            for viewer in self._viewers:
                viewer.it = (viewer.it - 1) % viewer.nt
        elif event.key == "right":
            for viewer in self._viewers:
                viewer.it = (viewer.it + 1) % viewer.nt
        elif event.key == "down":
            for viewer in self._viewers:
                viewer.ig = (viewer.ig - 1) % viewer.ng
        elif event.key == "up":
            for viewer in self._viewers:
                viewer.ig = (viewer.ig + 1) % viewer.ng

    def connect_key_events(self):
        for viewer in self._viewers:
            viewer.fig_i.canvas.mpl_connect("key_press_event", self.on_key_press)
            viewer.fig_j.canvas.mpl_connect("key_press_event", self.on_key_press)
            viewer.fig_k.canvas.mpl_connect("key_press_event", self.on_key_press)

    def savefigs(self, base: str, **kwargs):
        for i, viewer in enumerate(self._viewers):
            viewer.savefigs(f"{base}_{i}", **kwargs)
