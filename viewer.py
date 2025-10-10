import numpy as np
import array_api_compat.torch as xp
import matplotlib.pyplot as plt

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
            **self._kwargs,
            aspect=self._aspect_ratio_k,
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

            for axx in [self._ax_i, self._ax_j, self._ax_k]:
                axx.set_xticks([])
                axx.set_yticks([])

        if add_colorbar:
            cbar_i = self._fig_i.colorbar(self._img_i, ax=self._ax_i, **cbar_kws)
            cbar_j = self._fig_j.colorbar(self._img_j, ax=self._ax_j, **cbar_kws)
            cbar_k = self._fig_k.colorbar(self._img_k, ax=self._ax_k, **cbar_kws)
            cbar_i.ax.tick_params(labelsize="small")
            cbar_j.ax.tick_params(labelsize="small")
            cbar_k.ax.tick_params(labelsize="small")

        self._fig_i.show()
        self._fig_j.show()
        self._fig_k.show()

        # add interactivity
        self.connect_scroll_events()
        self.connect_key_events()

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

    def disconnect_scroll_events(self):
        if hasattr(self, "_cid_scroll_i") and hasattr(self, "_fig_i"):
            self._fig_i.canvas.mpl_disconnect(self._cid_scroll_i)
        if hasattr(self, "_cid_scroll_j") and hasattr(self, "_fig_j"):
            self._fig_j.canvas.mpl_disconnect(self._cid_scroll_j)
        if hasattr(self, "_cid_scroll_k") and hasattr(self, "_fig_k"):
            self._fig_k.canvas.mpl_disconnect(self._cid_scroll_k)

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

        self.connect_scroll_events()
        self.connect_key_events()

    def __del__(self):
        for viewer in self._viewers:
            viewer.connect_scroll_events()
            viewer.connect_key_events()

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
