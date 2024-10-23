
def get_mask(data):
    # Obtain binary mask from data
    masks = (data.notnull()) & (data > 0)

    # Convert the boolean mask to an integer mask (0 or 1)
    masks = masks.astype(bool)
    # Set the mask to nan where values are zeros
    masks = masks.where(data.notnull())
    masks = masks.assign_coords(coords=data.coords)
    masks.attrs = data.attrs
    
    return masks