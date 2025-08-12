class Node():
    def __init__(self, x):
        self.x = x
        self.next_node = None

def make_node_list(raw_data):
    start_node = Node(None)
    last_node = start_node
    for x in raw_data:
        temp_node = Node(x)
        last_node.next_node = temp_node

        last_node = temp_node
    return start_node

def print_node_list(node_list):
    node_list = node_list.next_node
    while node_list:
        print(node_list.x)
        node_list = node_list.next_node

def get_reverse_node_list(node_list):
    head = node_list
    current = head.next_node

    head.next_node = None
    while current:
        next_node = current.next_node
        current.next_node = head.next_node
        head.next_node = current
        current = next_node
    
    return head
    
if __name__ == "__main__":
    raw_data = [1, 2, 3]
    source_node_list = make_node_list(raw_data)
    reverse_node_list = get_reverse_node_list(source_node_list)
    print_node_list(reverse_node_list)


