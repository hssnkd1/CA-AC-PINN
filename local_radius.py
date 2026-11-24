"""Initial-condition-based local radius assignment; no final-time target used."""
import numpy as np

def initial_gradient_radii(dictionary, initial_value, strength=0.2, base_scale=1.5,
                           min_scale=0.8,max_scale=2.2,probe=0.003):
    """Adapt support size to gradients of GIVEN initial condition.

    Gradients are approximated by centered differences. The same normalized
    radius map applies at every level. No exact final solution enters selection.
    """
    if not (0 <= strength <= 1 and min_scale > 0 and max_scale >= min_scale):
        raise ValueError('Invalid radius configuration')
    z = dictionary.centers
    g2 = np.zeros(len(z))
    for j in range(dictionary.dim):
        plus=z.copy();minus=z.copy()
        plus[:,j]=np.minimum(z[:,j]+probe,1.)
        minus[:,j]=np.maximum(z[:,j]-probe,0.)
        grad=(initial_value(plus)-initial_value(minus))/(plus[:,j]-minus[:,j])
        g2 += grad**2
    mag=np.sqrt(g2)
    ref=max(float(np.quantile(mag, .90)), 1e-12)
    indicator=np.clip(mag/ref,0,1)
    scales=np.clip(base_scale*(1+strength*(.5-indicator)),min_scale,max_scale)
    dictionary.radii*=scales
    return {'min_scale':float(scales.min()),'max_scale':float(scales.max()),
            'mean_scale':float(scales.mean()),'indicator_p90':float(np.quantile(indicator,.9))}
