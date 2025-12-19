# plot
import matplotlib.pyplot as plt
import numpy as np

def plot_compare_estimators(
        k_fft, mean_fft, err_fft, binned_th_fft,
        k_dct, mean_dct, err_dct, binned_th_dct,
        theory_label=r'${\rm EoR\,Model}$',
        fft_label=r'${\rm Delay\,spectrum\,squared}$',
        dct_label=r'${\rm Fourier\, transform \, of \, C_\ell(\Delta\nu)}$',
        fft_dev_label=r'\rm{FFT\ deviation\ (\%)}',      # NEW
        dct_dev_label=r'\rm{DCT\ deviation\ (\%)}',      # NEW
        figsize=(7, 8),
        title=None, save=True,
        figname='compare_estimators.pdf',
        fft_color='C0', dct_color='C1',
        band=True, band_alpha=0.15,
        ylim=None, clip_symbol=True, verticle=False, vline=0.2):

    fig = plt.figure(figsize=figsize)

    # ----------------------------
    # NEW: 3 rows: top, FFT deviation, DCT deviation
    # ----------------------------
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.07)

    # ============================================================
    # TOP PANEL — Power spectra
    # ============================================================
    ax1 = fig.add_subplot(gs[0])

    fft_line = ax1.errorbar(k_fft, np.abs(mean_fft), err_fft, color=fft_color,
                            fmt='o', ms=4, elinewidth=1, ls='--', label=fft_label)          
    fft_color = fft_line[0].get_color()

    dct_line = ax1.errorbar(k_dct, np.abs(mean_dct), err_dct, color=dct_color,
                            fmt='s', ms=4, elinewidth=1, ls='--', label=dct_label)
    dct_color = dct_line[0].get_color()

    ax1.plot(k_fft, binned_th_fft, '-k', lw=2, label=theory_label)
    ax1.plot(k_dct, binned_th_dct, '-k', lw=2)

    ax1.set_yscale('log')
    ax1.set_ylabel(r'$P(k_\parallel)\,{\rm mK}^2\, {\rm Mpc}^3$')
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.tick_params(labelbottom=False)

    if title:
        ax1.set_title(title)
        # ax1.set_title(title, x=0.1, y=0.9)
        # y=0.98

    # ============================================================
    # Prepare ratios (same as before)
    # ============================================================
    ratio_fft = 100 * (mean_fft / binned_th_fft - 1)
    ratio_dct = 100 * (mean_dct / binned_th_dct - 1)

    err_fft_scaled = 100 * err_fft / binned_th_fft
    err_dct_scaled = 100 * err_dct / binned_th_dct

    # ===== Helper for clipped symbols =====
    def plot_clipped(ax, k, y, dy, marker, label, color=None):
        if ylim is None:
            ax.errorbar(k, y, dy, fmt=marker, capsize=1,  ms=4, elinewidth=1,
                        label=label, color=color)
            return

        mask_in = np.abs(y) <= ylim
        mask_up = y > ylim
        mask_down = y < -ylim

        if np.any(mask_in):
            line = ax.errorbar(k[mask_in], y[mask_in], dy[mask_in],
                               fmt=marker, capsize=1,  ms=4, elinewidth=1, label=label,
                               color=color)
            c = line[0].get_color()
        else:
            c = color

        if clip_symbol:
            if np.any(mask_up):
                for xi in k[mask_up]:
                    ax.text(xi, ylim * 0.8, r'$\uparrow$', ha='center',
                            va='bottom', fontsize=16, color=c)
            if np.any(mask_down):
                for xi in k[mask_down]:
                    ax.text(xi, -ylim * 0.8, r'$\downarrow$', ha='center',
                            va='top', fontsize=16, color=c)

    # ============================================================
    # MIDDLE PANEL — FFT percentage deviation
    # ============================================================
    ax_fft = fig.add_subplot(gs[1], sharex=ax1)    # NEW

    plot_clipped(ax_fft, k_fft, ratio_fft, err_fft_scaled,
                 'o', fft_label, color=fft_color)

    if band:
        ax_fft.fill_between(k_fft, -err_fft_scaled, err_fft_scaled,
                            color=fft_color, alpha=band_alpha,
                            label="_nolegend_")

    ax_fft.axhline(0, color='k', ls='--')
    # ax_fft.set_ylabel(r'\rm{FFT deviation (\%)}')
    ax_fft.set_ylabel(fft_dev_label)     # NEW
    ax_fft.grid(alpha=0.3)
    ax_fft.tick_params(labelbottom=False)

    if ylim is not None:
        ax_fft.set_ylim(-ylim, ylim)

    # ============================================================
    # BOTTOM PANEL — DCT percentage deviation
    # ============================================================
    ax_dct = fig.add_subplot(gs[2], sharex=ax1)    # NEW

    plot_clipped(ax_dct, k_dct, ratio_dct, err_dct_scaled,
                 's', dct_label, color=dct_color)

    if band:
        ax_dct.fill_between(k_dct, -err_dct_scaled, err_dct_scaled,
                            color=dct_color, alpha=band_alpha,
                            label="_nolegend_")

    ax_dct.axhline(0, color='k', ls='--')
    # ax_dct.set_ylabel(r'\rm{DCT deviation (\%)}')
    ax_dct.set_ylabel(dct_dev_label)     # NEW
    ax_dct.grid(alpha=0.3)

    if ylim is not None:
        ax_dct.set_ylim(-ylim, ylim)

    ax_dct.set_xlabel(r'$k_\parallel\,{\rm Mpc}^{-1}$')

    if (verticle==True):
        ax1.axvline(vline, ls='--', color='gray')
        ax_dct.axvline(vline, ls='--', color='gray')
        ax_fft.axvline(vline, ls='--', color='gray')
        

    fig.align_ylabels()

    if save:
        fig.savefig(figname, bbox_inches='tight')
        print("Saved:", figname)

    return fig, (ax1, ax_fft, ax_dct)

