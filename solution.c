/**
 * environment: 
 * gcc 11.2.0
**/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
//#include <omp.h>
#include <string.h>

#define TRUE 1
#define FALSE 0

#define NUM_CONFIGURATION 2


// Node of linked list
typedef struct queue_node{
    int node;
    struct queue_node *next;
}Queue_Node;

// Priority Queue
typedef struct queue{
    Queue_Node *head;
    Queue_Node *last;
    unsigned int length;
}Queue;

// Coordinate
typedef struct coordinate{
    int x;
    int y;
}Coordinate;





// return the minimum of two
int min(int a, int b);

//===========================================================
//adjacent matrix
/**
 * read adjacent matrix from input
 * return adjacent matrix
 */
int **read_adjacent_matrix(int *node_number, int *edge_number);
/**
 * copy an adjacent matrix
 */
int **copy_adjacent_matrix(int node_number,int** adjacent_matrix);
/**
 * create empty adjacent matrix
 */
int **create_adjacent_matrix(int node_number);
//free the adjacent matrix
void free_adjacent_matrix(int node_number, int** adjacent_matrix);
//print the adjacent matrix to stdout
void print_adjacent_matrix(int node_number, int** adjacent_matrix);

//===========================================================
//queue

// create an empty priority queue
Queue *create_queue();
// create a new node
Queue_Node *new_node(int node);
// whether the queue is empty
int isEmpty(Queue *queue);
// free a priority queue and data inside
void free_queue(Queue *queue);
// push a node to the tail of queue
void push(Queue *queue, int node);
// pop the first element and return node
int pop(Queue *queue);

//===========================================================
//sequential algorithm
void sequential_vertex_cover(int node_number,int edge_number, int** adjacent_matrix);


//===========================================================
//vertex cover algorithm
/**
 * find the vertex cover (return null if not found)
*/
int* recursive_find_vertex_cover(int node_number, Coordinate *edges, 
                                int edge_number,  int **adjacent_matrix,
                                int **edge_covered, 
                                int current_subset_node, 
                                int subset_node_number, int *vertex_subset);
/**
 * add the last node to the subset and verify 
*/
int* last_verify(int node_number, Coordinate *edges, 
                int edge_number,  int **adjacent_matrix,
                int **edge_covered, 
                int current_subset_node, 
                int subset_node_number, int *vertex_subset);
/**
 * verify whether the input matrix cover every node
*/
int verify(Coordinate *edges, int edge_number, int **edge_covered);
/**
 * update the covered edge
*/
void update_covered_edge(int added_node,int node_number, int** adjacent_matrix, int **edge_covered);


// read
int main(int argc, char * argv[]){

    int node_number,edge_number,is_parallel,number_thread;
    int **adjacent_matrix;
    
    // if(argc != NUM_CONFIGURATION + 1){
    //     perror("Not enough configuration!");
    //     exit(1);
    // }

    // is_parallel = atoi(argv[1]);
    // number_thread = atoi(argv[2]);

    //scan the input
    adjacent_matrix = read_adjacent_matrix(&node_number,&edge_number);

    //print number of node and edge
    printf("%d %d ",node_number,edge_number);

    // set number of threads
    // omp_set_num_threads(number_thread);

    // computation
    // if(is_parallel == 0){
    //     sequential_vertex_cover(node_number,edge_number,adjacent_matrix);
    // }else{
    //     perror("is_parallel wrong!");
    //     exit(1);
    // }

    sequential_vertex_cover(node_number,edge_number, adjacent_matrix);

    //free the matrix
    free_adjacent_matrix(node_number,adjacent_matrix);

    return 0;

}

// return the minimum of two
int min(int a, int b){
    if(a < b){
        return a;
    }
    return b;
}

/**
 * read adjacent matrix from input
 * return adjacent matrix
 */
int **read_adjacent_matrix(int *node_number, int *edge_number){
    // store input edge u->v with capactiy
    int u,v,current_edge_number = 0;
    int **adjacent_matrix;
    int i;

    //scan the number of node
    scanf("%d\n", node_number);

    adjacent_matrix = create_adjacent_matrix(*node_number);

    //read the edges
    while(scanf("%d %d\n", &u,&v) == 2){
        adjacent_matrix[u][v] = 1;
        adjacent_matrix[v][u] = 1;
        current_edge_number++;
    }

    *edge_number = current_edge_number;

    return adjacent_matrix;

}

/**
 * copy an adjacent matrix
 */
int **copy_adjacent_matrix(int node_number,int** adjacent_matrix){
    int i,j;
    int **new_adjacent_matrix;

    //allocate the adjacent matrix (use malloc to save time)
    new_adjacent_matrix = (int **)(malloc(sizeof(int*) * (node_number)));
    for(i=0;i<node_number;i++){
        new_adjacent_matrix[i] = (int *)(malloc(node_number *sizeof(int)));
    }

    for(i=0;i<node_number;i++){
        for(j=0;j<node_number;j++){
            new_adjacent_matrix[i][j] = adjacent_matrix[i][j];
        }
    }

    return new_adjacent_matrix;
}

/**
 * create empty adjacent matrix
 */
int **create_adjacent_matrix(int node_number){
    int i;
    int **adjacent_matrix;

    //allocate the adjacent matrix
    adjacent_matrix = (int **)(malloc(sizeof(int*) * (node_number)));
    for(i=0;i<node_number;i++){
        adjacent_matrix[i] = (int *)(calloc(node_number,sizeof(int)));
    }

    return adjacent_matrix;

}



//free the adjacent matrix
void free_adjacent_matrix(int node_number, int** adjacent_matrix){

    int i;
    for(i=0;i<node_number;i++){
        free(adjacent_matrix[i]);
    }
    free(adjacent_matrix);

}

//print the adjacent matrix to stdout
void print_adjacent_matrix(int node_number, int** adjacent_matrix){

    int i,j;
    for(i=0;i<node_number;i++){
        for(j=0;j<node_number;j++){
            printf("%d ",adjacent_matrix[i][j]);
        }
        printf("\n");
    }

}

// create an empty priority queue
Queue *create_queue(){
    Queue *temp = (Queue *)malloc(sizeof(Queue));
    temp->head = NULL;
    temp->last = NULL;
    temp->length = 0;
    return temp;
}

// create a new node
Queue_Node *new_node(int node){
    Queue_Node *temp = (Queue_Node *)malloc(sizeof(Queue_Node));
    temp->node = node;
    temp->next = NULL;
 
    return temp;
}

// whether the queue is empty
int isEmpty(Queue *queue){
    return queue->length == 0;
}

// free a priority queue and data inside
void free_queue(Queue *queue){
    if(queue == NULL){
        return;
    }
    Queue_Node *curr_node=queue->head;
    Queue_Node *temp_node;
    while(curr_node!=NULL){
        temp_node = curr_node;
        curr_node = curr_node->next;
        free(temp_node);
    }
    free(queue);
}

// push a node to the tail of queue
void push(Queue *queue, int node){

    Queue_Node *Qnode;

    //create new queue node
    Qnode = new_node(node);
    
    if(queue->head == NULL){
        //if it is the first node
        queue->head = Qnode;
        queue->last = Qnode;
    }else{
        //add it to the tail of queue
        queue->last->next = Qnode;
        queue->last = Qnode;
    }
    (queue->length)++;

}

// pop the first element and return node
int pop(Queue *queue){

    int node;

    if(queue->head==NULL){
        printf("ERROR: try to pop from an empty queue!");
        return -1;
    }

    Queue_Node *temp = queue->head;
    node = temp->node;
    queue->head = queue->head->next;
    free(temp);
    (queue->length)--;

    //queue is empty
    if(queue->head == NULL){
        queue->last = NULL;
    }
    
    return node;
}


//find the minimal set of vertices that cover all the edges
void sequential_vertex_cover(int node_number,int edge_number, int** adjacent_matrix){
    //number of vertices in the set
    int number_vertices;
    int i,j,k,current_edges=0;
    // the valid vertex cover (might be NULL)
    int *vertex_cover_subset = NULL;

    //coordinate of edge
    Coordinate temp_coord;

    //record the current vertex set
    int *vertex_set = (int*)malloc(sizeof(int) * node_number);
    //a 2d matrix which record the edge touched by vertices
    int **edge_covered = create_adjacent_matrix(node_number);

    //store the coordinates of edges into a dynamic array
    Coordinate *edges = (Coordinate *)malloc(sizeof(Coordinate) * edge_number);
    for(i=0;i<node_number;i++){
        for(j=i;j<node_number;j++){
            if(adjacent_matrix[i][j] != 0){
                temp_coord.x = i;
                temp_coord.y = j;
                edges[current_edges] = temp_coord;
                current_edges++;
            }
        }
    }

    // loop throught all the vertices set to find minimal vertex cover
    for(number_vertices=1;number_vertices<=node_number;number_vertices++){

        vertex_cover_subset = recursive_find_vertex_cover(node_number, 
                                edges, edge_number, adjacent_matrix,
                                edge_covered, 0, number_vertices, NULL);

        // stop when find valid vertex cover
        if(vertex_cover_subset != NULL){
            break;
        }

    }


    // print the result
    printf("%d ", number_vertices);

    for(i=0;i<number_vertices;i++){
        printf("%d ", vertex_cover_subset[i]);
    }

    //free the dynamic resources
    free(edges);
}


/**
 * find the vertex cover (return null if not found)
*/
int* recursive_find_vertex_cover(int node_number, Coordinate *edges, 
                                int edge_number,  int **adjacent_matrix,
                                int **edge_covered, 
                                int current_subset_node, 
                                int subset_node_number, int *vertex_subset){
    int i;
    int *current_vertex_subset, **current_edge_covered;
    // temp vertex cover (might be NULL)
    int *temp_vertex_cover_subset;
    // the valid vertex cover
    int *vertex_cover_subset = NULL;

    // number of vertices which needed to be add to the subset
    int node_left = subset_node_number - current_subset_node;

    // base case
    if(node_left == 1){
        return last_verify(node_number, edges, edge_number,  adjacent_matrix,
                        edge_covered, current_subset_node, subset_node_number, 
                        vertex_subset);
    }

    // the start number of the for loop
    int range_left;
    //first vertex to be add to the subset
    if(current_subset_node == 0){
        range_left = 0;
    }else{
        range_left = vertex_subset[current_subset_node-1] + 1;
    }
    // the end number of the for loop
    int range_right = node_number - subset_node_number + current_subset_node + 1;

    for(i=range_left;i<range_right;i++){
        // prepare the data for next recursion
        current_edge_covered = copy_adjacent_matrix(node_number, edge_covered);
        update_covered_edge(i,node_number,adjacent_matrix,current_edge_covered);

        current_vertex_subset = (int *)malloc(sizeof(int) * (current_subset_node + 1));
        memcpy(current_vertex_subset, vertex_subset, sizeof(int) * (current_subset_node));
        current_vertex_subset[current_subset_node] = i;

        temp_vertex_cover_subset = recursive_find_vertex_cover(node_number, 
                                edges, edge_number, adjacent_matrix,
                                current_edge_covered, current_subset_node+1, 
                                subset_node_number, current_vertex_subset);

        if(temp_vertex_cover_subset != NULL){
            vertex_cover_subset = temp_vertex_cover_subset;
        }
        
        // free the temp data
        free_adjacent_matrix(node_number, current_edge_covered);
        free(current_vertex_subset);

    }

    return vertex_cover_subset;

}

/**
 * add the last node to the subset and verify 
*/
int* last_verify(int node_number, Coordinate *edges, 
                int edge_number,  int **adjacent_matrix,
                int **edge_covered, 
                int current_subset_node, 
                int subset_node_number, int *vertex_subset){

    int i,j,k;
    int *current_vertex_subset, **current_edge_covered;
    // temp vertex cover (might be NULL)
    int *temp_vertex_cover_subset;
    // the valid vertex cover
    int *vertex_cover_subset = NULL;

    // the start number of the for loop
    int range_left;
    //first vertex to be add to the subset
    if(current_subset_node == 0){
        range_left = 0;
    }else{
        range_left = vertex_subset[current_subset_node-1] + 1;
    }
    // the end number of the for loop
    int range_right = node_number - subset_node_number + current_subset_node + 1;

    // prepare the matrix for edge covered by subset
    current_edge_covered = create_adjacent_matrix(node_number);

    for(i=range_left;i<range_right;i++){

        //copy data from edge_covered
        for(j=0;j<node_number;j++){
            for(k=0;k<node_number;k++){
                current_edge_covered[j][k] = edge_covered[j][k];
            }
        }

        update_covered_edge(i,node_number,adjacent_matrix,current_edge_covered);

        // find valid vertex cover
        if(verify(edges, edge_number, current_edge_covered)){

            // construct the subset for valid vertex cover
            current_vertex_subset = (int *)malloc(sizeof(int) * (current_subset_node + 1));
            memcpy(current_vertex_subset, vertex_subset, sizeof(int) * (current_subset_node));
            current_vertex_subset[current_subset_node] = i;

            // release the resource
            free_adjacent_matrix(node_number, current_edge_covered);

            return current_vertex_subset;
        }
    }

    // release the resource
    free_adjacent_matrix(node_number, current_edge_covered);

    // did not found any valid vertex cover
    return NULL;

}

/**
 * verify whether the input matrix cover every node
*/
int verify(Coordinate *edges, int edge_number, int **edge_covered){
    int i,x,y;

    // check whether each edge is covered
    for(i=0;i<edge_number;i++){
        x = edges[i].x;
        y = edges[i].y;
        //not covered
        if(!edge_covered[x][y]){
            return FALSE;
        }
    }

    return TRUE;

}

/**
 * update the covered edge
*/
void update_covered_edge(int added_node,int node_number, int** adjacent_matrix, int **edge_covered){
    int i;
    for(i=0;i<node_number;i++){
        edge_covered[i][added_node] = edge_covered[i][added_node] || adjacent_matrix[i][added_node];
        edge_covered[added_node][i] = edge_covered[added_node][i] || adjacent_matrix[added_node][i];
    }

}