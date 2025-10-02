import torch
import numpy as np
import matplotlib.pyplot as plt


section_info_dim = 2
Myield_start_index = 24
Myield_end_index = 24 + (6 - 1) * section_info_dim + 1
Myield_yn_index = 28
Myield_yp_index = 30
output_start_index = 6
output_end_index = 12 
output_yn_index = 8
output_yp_index = 9
Hz = 20
yield_factor = 0.9



def visualize_plasticHinge(graph, save_dir):

    # Plot
    original_x = graph.x[:, :6]
    x_grid_num, y_grid_num, z_grid_num = graph.x[0, 0:3].cpu().numpy().astype(int) 
    
        
    for z in range(z_grid_num):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        # fig.suptitle(f"Plastic Hinge Visualization --- Z{z} Section", fontsize=19, fontweight='bold')

        for y in range(y_grid_num):
            for x in range(x_grid_num):
                grid_coord = np.array([x, y, z])                
                node_index = 0
                for i in range(graph.x.shape[0]):
                    if (original_x[i, 3:6].cpu().numpy() == grid_coord).all():
                        node_index = i
                        break

                # Plot structure skeleton -- beam (0F no beam)
                if x != x_grid_num - 1 and y != 0:
                    # Get x_p section My
                    My_x_p = graph.x[node_index, Myield_start_index + 1 * section_info_dim].cpu().numpy() / 2552083.0
                    # print(f"Node num: {node_index+1},", f"{x} {y} {z}, ", graph.x[node_index, classifier.Myield_start_index:classifier.Myield_start_index+6])
                    # color = 1 - (My_x_p - 0.3) / 0.7 * 0.8
                    color = My_x_p
                    ax.plot([x, x+1], [y, y], linewidth=5, color=(color, color, color), marker='o', markerfacecolor='k', markersize=10, zorder=1)


        for x in range(x_grid_num):   
            for y in range(y_grid_num):   
                grid_coord = np.array([x, y, z])            
                node_index = 0
                for i in range(graph.x.shape[0]):
                    if (original_x[i, 3:6].cpu().numpy() == grid_coord).all():
                        node_index = i
                        break

                # Plot structure skeleton -- column
                if y != y_grid_num - 1:
                    # Get y_p section My
                    My_y_p = graph.x[node_index, Myield_start_index + 3 * section_info_dim].cpu().numpy() / 2552083.0
                    # color = 1 - (My_y_p - 0.3) / 0.7 * 0.8
                    color = My_x_p
                    ax.plot([x, x], [y, y+1], linewidth=5, color=(color, color, color), marker='o', markerfacecolor='k', markersize=10, zorder=1)
                    
                
                # Plot plastic hinge
                for i, face_index in enumerate(list(range(2, 8))):   # face_index is index for Mz(x_n, x_p, y_n, y_p, z_n, z_p)
                    Myield_face_i = graph.x[node_index, Myield_start_index + i * section_info_dim]
                    if Myield_face_i <= 0.1:     # It means this face is not connect to any element.
                        continue
                    real_node_plastic_hinge = torch.max(graph.y[node_index, :, face_index].abs()) >= yield_factor * Myield_face_i

                   
                    if real_node_plastic_hinge:
                        if i == 0:
                            ax.add_artist(plt.Circle((x - 0.2, y), 0.05, fill=True, color='red'))
                        elif i == 1:
                            ax.add_artist(plt.Circle((x + 0.2, y), 0.05, fill=True, color='red'))
                        elif i == 2:
                            ax.add_artist(plt.Circle((x, y - 0.2), 0.05, fill=True, color='red'))
                        elif i == 3:
                            ax.add_artist(plt.Circle((x, y + 0.2), 0.05, fill=True, color='red'))


        ax.set_xlabel('x axis (mm)')
        ax.set_ylabel('y axis (mm)')
        ax.set_xlim((-1, x_grid_num))
        ax.set_title("TRUE")

        plt.savefig(save_dir / f"Z{z}.png")
        plt.close()
        # plt.show()
        
        
        
    for x in range(x_grid_num):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        # fig.suptitle(f"Plastic Hinge Visualization --- X{x} Section", fontsize=19, fontweight='bold')

        for y in range(y_grid_num):
            for z in range(z_grid_num):
                grid_coord = np.array([x, y, z])                
                node_index = 0
                for i in range(graph.x.shape[0]):
                    if (original_x[i, 3:6].cpu().numpy() == grid_coord).all():
                        node_index = i
                        break

                # Plot structure skeleton -- beam (0F no beam)
                if x != x_grid_num - 1 and y != 0:
                    # Get x_p section My
                    My_x_p = graph.x[node_index, Myield_start_index + 1 * section_info_dim].cpu().numpy() / 2552083.0
                    # print(f"Node num: {node_index+1},", f"{x} {y} {z}, ", graph.x[node_index, classifier.Myield_start_index:classifier.Myield_start_index+6])
                    # color = 1 - (My_x_p - 0.3) / 0.7 * 0.8
                    color = My_x_p
                    ax.plot([x, x+1], [y, y], linewidth=5, color=(color, color, color), marker='o', markerfacecolor='k', markersize=10, zorder=1)


        for x in range(x_grid_num):   
            for y in range(y_grid_num):   
                grid_coord = np.array([x, y, z])            
                node_index = 0
                for i in range(graph.x.shape[0]):
                    if (original_x[i, 3:6].cpu().numpy() == grid_coord).all():
                        node_index = i
                        break

                # Plot structure skeleton -- column
                if y != y_grid_num - 1:
                    # Get y_p section My
                    My_y_p = graph.x[node_index, Myield_start_index + 3 * section_info_dim].cpu().numpy() / 2552083.0
                    # color = 1 - (My_y_p - 0.3) / 0.7 * 0.8
                    color = My_x_p
                    ax.plot([x, x], [y, y+1], linewidth=5, color=(color, color, color), marker='o', markerfacecolor='k', markersize=10, zorder=1)
                    
                
                # Plot plastic hinge
                for i, face_index in enumerate(list(range(2, 8))):   # face_index is index for Mz(x_n, x_p, y_n, y_p, z_n, z_p)
                    Myield_face_i = graph.x[node_index, Myield_start_index + i * section_info_dim]
                    if Myield_face_i <= 0.1:     # It means this face is not connect to any element.
                        continue
                    real_node_plastic_hinge = torch.max(graph.y[node_index, :, face_index].abs()) >= yield_factor * Myield_face_i

                   
                    if real_node_plastic_hinge:
                        if i == 0:
                            ax.add_artist(plt.Circle((x - 0.2, y), 0.05, fill=True, color='red'))
                        elif i == 1:
                            ax.add_artist(plt.Circle((x + 0.2, y), 0.05, fill=True, color='red'))
                        elif i == 2:
                            ax.add_artist(plt.Circle((x, y - 0.2), 0.05, fill=True, color='red'))
                        elif i == 3:
                            ax.add_artist(plt.Circle((x, y + 0.2), 0.05, fill=True, color='red'))


        ax.set_xlabel('x axis (mm)')
        ax.set_ylabel('y axis (mm)')
        ax.set_xlim((-1, x_grid_num))
        ax.set_title("TRUE")

        plt.savefig(save_dir / f"X{x}.png")
        plt.close()