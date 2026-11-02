"""
Prototype test script for improved HELLO, C, and NO definitions.
"""
import numpy as np

def create_base_hand():
    return {
        0:  np.array([0.50, 0.85, 0.00]), # Wrist
        1:  np.array([0.43, 0.76, -0.01]),
        2:  np.array([0.37, 0.67, -0.02]),
        3:  np.array([0.33, 0.58, -0.03]),
        4:  np.array([0.30, 0.50, -0.04]),
        5:  np.array([0.44, 0.56, -0.01]),
        6:  np.array([0.43, 0.44, -0.01]),
        7:  np.array([0.42, 0.35, -0.01]),
        8:  np.array([0.41, 0.25, -0.01]),
        9:  np.array([0.50, 0.54, 0.00]),
        10: np.array([0.50, 0.41, 0.00]),
        11: np.array([0.50, 0.31, 0.00]),
        12: np.array([0.50, 0.21, 0.00]),
        13: np.array([0.56, 0.56, 0.01]),
        14: np.array([0.57, 0.44, 0.01]),
        15: np.array([0.58, 0.35, 0.01]),
        16: np.array([0.59, 0.26, 0.01]),
        17: np.array([0.62, 0.60, 0.02]),
        18: np.array([0.64, 0.50, 0.02]),
        19: np.array([0.65, 0.43, 0.02]),
        20: np.array([0.66, 0.35, 0.02])
    }

def get_new_hello():
    """HELLO: Open hand with fingers spread wide in a greeting / wave posture."""
    hand = create_base_hand()
    # Thumb spread far out to left
    hand[1] = np.array([0.41, 0.76, -0.02])
    hand[2] = np.array([0.33, 0.68, -0.03])
    hand[3] = np.array([0.26, 0.60, -0.04])
    hand[4] = np.array([0.20, 0.52, -0.04]) # Thumb wide out
    
    # Fingers spread apart like a fan
    # Index spread to left
    hand[6] = np.array([0.39, 0.43, -0.01])
    hand[7] = np.array([0.36, 0.32, -0.01])
    hand[8] = np.array([0.33, 0.21, -0.01]) # Index spread left
    
    # Middle straight up
    hand[10] = np.array([0.48, 0.40, 0.0])
    hand[11] = np.array([0.48, 0.28, 0.0])
    hand[12] = np.array([0.48, 0.17, 0.0]) # Middle up
    
    # Ring spread right
    hand[14] = np.array([0.58, 0.42, 0.01])
    hand[15] = np.array([0.62, 0.31, 0.01])
    hand[16] = np.array([0.65, 0.20, 0.01]) # Ring spread right
    
    # Pinky spread far right
    hand[18] = np.array([0.67, 0.49, 0.02])
    hand[19] = np.array([0.72, 0.40, 0.02])
    hand[20] = np.array([0.77, 0.30, 0.02]) # Pinky spread far right
    return hand

def get_new_stop():
    """STOP: Rigid flat palm, fingers straight up and held tightly together."""
    hand = create_base_hand()
    # Thumb resting slightly out
    hand[4] = np.array([0.32, 0.52, -0.02])
    # 4 fingers tightly together pointing vertically up
    hand[6] = np.array([0.44, 0.42, 0.0])
    hand[7] = np.array([0.44, 0.30, 0.0])
    hand[8] = np.array([0.44, 0.19, 0.0]) # Index
    
    hand[10] = np.array([0.48, 0.40, 0.0])
    hand[11] = np.array([0.48, 0.28, 0.0])
    hand[12] = np.array([0.48, 0.17, 0.0]) # Middle
    
    hand[14] = np.array([0.52, 0.41, 0.0])
    hand[15] = np.array([0.52, 0.29, 0.0])
    hand[16] = np.array([0.52, 0.18, 0.0]) # Ring
    
    hand[18] = np.array([0.56, 0.44, 0.0])
    hand[19] = np.array([0.56, 0.33, 0.0])
    hand[20] = np.array([0.56, 0.23, 0.0]) # Pinky
    return hand

def get_new_c(facing="left"):
    """C: Curved hand forming a clear 'C' shape with distinct open gap between thumb and fingers."""
    hand = create_base_hand()
    mult = 1.0 if facing == "left" else -1.0
    center_x = 0.50
    
    # Thumb curving up and in
    hand[1] = np.array([center_x - 0.04*mult, 0.76, 0.0])
    hand[2] = np.array([center_x - 0.08*mult, 0.70, 0.0])
    hand[3] = np.array([center_x - 0.12*mult, 0.65, 0.0])
    hand[4] = np.array([center_x - 0.10*mult, 0.58, 0.0]) # Thumb tip lower rim of C
    
    # 4 fingers curved forward/down forming upper rim of C
    for base_idx, mcp_x, tip_x in [(5, center_x-0.03*mult, center_x-0.10*mult),
                                    (9, center_x, center_x-0.08*mult),
                                    (13, center_x+0.03*mult, center_x-0.06*mult),
                                    (17, center_x+0.06*mult, center_x-0.04*mult)]:
        hand[base_idx] = np.array([mcp_x, 0.54, 0.0])
        hand[base_idx+1] = np.array([mcp_x - 0.04*mult, 0.44, -0.02])
        hand[base_idx+2] = np.array([mcp_x - 0.07*mult, 0.36, -0.04])
        hand[base_idx+3] = np.array([tip_x, 0.34, -0.05]) # Fingertip upper rim of C
    return hand

def get_new_no(state="pinched"):
    """NO: Index and middle fingers extended with tips touching thumb tip; ring & pinky curled."""
    hand = create_base_hand()
    # Thumb pointing up-inward
    hand[1] = np.array([0.45, 0.75, 0.0])
    hand[2] = np.array([0.43, 0.66, 0.0])
    hand[3] = np.array([0.44, 0.57, 0.0])
    
    thumb_y = 0.48 if state == "pinched" else 0.52
    hand[4] = np.array([0.46, thumb_y, 0.02]) # Thumb tip
    
    # Index finger extended forward/up and curving to meet thumb
    idx_tip_y = 0.48 if state == "pinched" else 0.45
    hand[5] = np.array([0.44, 0.55, 0.0])
    hand[6] = np.array([0.43, 0.44, 0.0])
    hand[7] = np.array([0.44, 0.43, 0.01])
    hand[8] = np.array([0.46, idx_tip_y, 0.02]) # Index tip touches thumb
    
    # Middle finger extended forward/up and curving to meet thumb
    mid_tip_y = 0.48 if state == "pinched" else 0.46
    hand[9] = np.array([0.49, 0.54, 0.0])
    hand[10] = np.array([0.48, 0.43, 0.0])
    hand[11] = np.array([0.47, 0.43, 0.01])
    hand[12] = np.array([0.46, mid_tip_y, 0.02]) # Middle tip touches thumb
    
    # Ring and pinky tightly curled into palm
    hand[13] = np.array([0.54, 0.56, 0.0])
    hand[14] = np.array([0.54, 0.62, 0.03])
    hand[15] = np.array([0.53, 0.68, 0.04])
    hand[16] = np.array([0.51, 0.71, 0.03])
    
    hand[17] = np.array([0.58, 0.59, 0.0])
    hand[18] = np.array([0.58, 0.64, 0.03])
    hand[19] = np.array([0.57, 0.70, 0.04])
    hand[20] = np.array([0.55, 0.73, 0.03])
    return hand

print("Prototypes defined successfully.")
