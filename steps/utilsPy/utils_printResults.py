import matplotlib.pyplot as plt

# Remove all the white space in the column names
def correct_column_names(list_columns): 
    columns_names_new = []
    for column in list_columns:
        column = column.replace(" ", "")
        columns_names_new.append(column)
    return columns_names_new

# Single line plot
def plot_single(values_list, name_graphic):
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(name_graphic)
    plt.plot(values_list)
    plt.legend(['train', 'val'], loc='upper right') 
    plt.savefig(f"plots/{name_graphic}.png")
    plt.close()

# Double line plot for train + validation
def plot_generic(values_list_train, values_list_val, name_graphic):
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(name_graphic)
    plt.plot(values_list_train)
    plt.plot(values_list_val)
    plt.legend(['train', 'val'], loc='upper right') 
    plt.savefig(f"plots/{name_graphic}.png")
    plt.close()

