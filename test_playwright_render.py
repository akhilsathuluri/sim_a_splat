# %%
from sim_a_splat.env.splat.splat_scara_env import SplatEnv
from pathlib import Path
import numpy as np
import imageio
from pydrake.all import RotationMatrix, RollPitchYaw
import time

# %%
package_path = Path(__file__).resolve().parent.parent / "sim_a_splat/robot_description/"
package_name = "scara/"
urdf_name = "scara.urdf"
eef_link_name = "link_3"
splat_env = SplatEnv(
    visualise_sim_flag=True,
    eef_link_name=eef_link_name,
    package_path=package_path,
    package_name=package_name,
    urdf_name=urdf_name,
)

# %% In a seperate terminal!!!
# Use Playwright to connect to the visualization server
print("Connecting to visualization server...")
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("http://localhost:8087")
    page.wait_for_load_state('networkidle')
    print("Connected to visualization server successfully")
    # time.sleep(3)
    # print("Resetting the environment...")
    # obs = splat_env.reset(seed=0)
    # img = splat_env.render()
    # output_file = "splat_render.png"
    # imageio.imwrite(output_file, img[0])
    # print(f"Rendered image saved to {output_file}")

# %%
print("Resetting the environment...")
obs = splat_env.reset(seed=0)



    #     page.wait_for_load_state("networkidle")
    #     print("Connected to visualization server successfully")

    #     # Wait for WebGL content to initialize
    #     time.sleep(3)

    #     # Render the environment
    #     print("Rendering the environment...")
    #     img = splat_env.render()

    #     # Save the rendered image
    #     output_file = "splat_render.png"
    #     imageio.imwrite(output_file, img)
    #     print(f"Rendered image saved to {output_file}")

    # except Exception as e:
    #     print(f"Error during rendering process: {e}")
    # finally:
    #     browser.close()

# %%
img = splat_env.render()
import matplotlib.pyplot as plt
plt.imshow(img[0])
plt.axis('off')
# %%
