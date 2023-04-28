#include <iostream>
#include <omp.h>
#include <vector>
#include <queue>

using namespace std;

class Graph
{
    int vertices;
    vector<vector<int>> graph;

public:
    Graph(int v)
    {
        vertices = v;
        graph.resize(v);
    }

    void addEdge(int m, int n)
    {
        graph[m].push_back(n);
        graph[n].push_back(m);
    }

    void sequential_bfs(int src)
    {
        vector<bool> visited(vertices, false);
        queue<int> q;

        visited[src] = true;
        q.push(src);

        while (!q.empty())
        {
            int node = q.front();
            q.pop();

            for (int ele : graph[node])
            {
                if (!visited[ele])
                {
                    visited[ele] = true;
                    q.push(ele);
                }
            }
        }
    }

    void parallel_bfs(int src)
    {
        int *visited = new int[vertices];
        for (int i = 0; i < vertices; i++)
        {
            visited[i] = 0;
        }

        // vector<int> visited(vertices, 0);
        queue<int> q;

        visited[src] = 1;
        q.push(src);

        while (!q.empty())
        {
            int node = q.front();
            q.pop();

#pragma omp parallel for
            for (int i = 0; i < graph[node].size(); i++)
            {
                int ele = graph[node][i];

                // cout << ele << " ";

                // Perform an atomic read to get the current value of visited[ele]
                int ele_visited = 0;
#pragma omp atomic read
                ele_visited = visited[ele];

                // Perform an atomic write to set visited[ele] to true if it's not already set
                if (!ele_visited)
                {
#pragma omp atomic write
                    visited[ele] = 1;
                    q.push(ele);
                }
            }
        }
    }
};

int main()
{
    int n;
    double t1, t2;

    cout << "Enter number of nodes : ";
    cin >> n;

    Graph g(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if ((i + 3 * j) % 7 == 0 || (j + 2 * i) % 5 == 0)
            {
                g.addEdge(i, j);
            }
        }
    }

    t1 = omp_get_wtime();
    g.sequential_bfs(0);
    t2 = omp_get_wtime();
    double timeTakenSerial = t2 - t1;
    cout << "Sequential BFS Time : " << timeTakenSerial << endl;

    t1 = omp_get_wtime();
    g.parallel_bfs(0);
    t2 = omp_get_wtime();
    double timeTakenParallel = t2 - t1;
    cout << "Parallel BFS Time : " << timeTakenParallel << endl;

    cout << "Speed Up : " << timeTakenSerial / timeTakenParallel << endl;
    return 0;
}
