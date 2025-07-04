# Dataset utilizado para el modelo CNN_mem.
# Para cada día se tiene, además, la indicación vállidos/inválidos
# para el día anterior.

memory_series = series.copy()
for i in range(len(memory_series)):
    if i == 0:
        memory_series[i].append([0.0,0.0,0.0])
    else:
        memory_series[i].append(series[i-1][1])

class dni_mem(Dataset):
    """Dataset de pytorch de las series temporales de radiación directa para
    el modelo CNN_mem

    Importante: este dataset es experimental. No se puede usar puesto que utiliza
    información de clasificación aportada por un humano.

    Attributes:
        root_dir (str): La ruta base donde se encuentran las series temporales
        files (list): lista de los nombres de los archivos de las series
            temporales. Debe estar en el mismo formato que el devuelto por
            reshape_series() y además tener cada día la información de validez del
            día anterior.
    """
    
    def __init__(self, root_dir, files):
        self.root_dir = root_dir
        self.files = files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        series = []
        item = self.files[index]
        for i in range(NUM_S):
            path = self.root_dir + SENSORS[i] + "/series" 
            path += ("/validas" if item[1][i] == 0 else "/no validas")
            path += "/" + item[0]
            series.append(get_points(path))
        return ([torch.tensor(series),torch.tensor(item[2])], torch.tensor(item[1]))
    

class CNN_mem(nn.Module):
    """Modelo de pytorch de aprendizaje profundo.

    Se trata de una red neuronal convolucional, similar a CNN. En este caso
    se utiliza también la información de la clasificación de la serie temporal en 
    válido/inválido del día anterior.

    """
    def __init__(self, num_sensors):
        super(CNN_mem,self).__init__()
        self.conv1 = nn.Conv2d(1,3,3)
        self.conv2 = nn.Conv2d(3,9,3)
        self.pool = nn.AvgPool2d((1,2))
        self.dense1 = nn.Linear(9*3*34+num_sensors,50)        #144-2=142
        self.dense2 = nn.Linear(50,num_sensors)             #142/2=71
        self.num_sensors = num_sensors            #71-2=69
                                                  #69/2=34
                                                  #9*3*34= 918
    def forward(self, x, mem):
        """Recorrido de un input por la red neuronal
        
        La secuencia es la siguiente:
        convolución -> pool -> convolución -> pool ->
        flatten -> concatenación con output del día anterior ->
        capa densa -> capa densa

        La función de activación empleada es la sigmoide.
        
        Args:
            x: matriz del input junto a la información de clasificación
                del día anterior.

        Returns:
            x: lista de tres elementos que indican la validez
                de las series temporales para cada sensor.
        """
        x = F.pad(x, (0, 0, 1, 1), mode='circular')
        x = self.pool(F.sigmoid(self.conv1(x)))
        x = F.pad(x, (0, 0, 1, 1), mode='circular')
        x = self.pool(F.sigmoid(self.conv2(x)))

        x = x.view(1,9*3*34)
        #x = self.flatten(x)
        # print(x.shape)
        # x = torch.squeeze(x)
        # print(x.shape)
        # print(mem.shape)
        mem = torch.reshape(mem.clone().detach(),(1,self.num_sensors))
        x = F.sigmoid(self.dense1(torch.cat((x,mem),dim=1)))
        x = self.dense2(x)
        return x
    
class ViT_mem(nn.Module):
    def __init__(self, *, series_length, patch_length, num_sensors, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        assert series_length % patch_length == 0, 'El tamaño de la serie temporal debe ser divisible por el tamaño del patch.'

        num_patches = series_length // patch_length
        patch_dim = num_sensors * patch_length
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b s (l p) -> b l (p s)', p = patch_length),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.dim = dim
        self.num_sensors = num_sensors

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(num_sensors + dim, num_sensors)
        self.mlp_head = nn.Sequential(
            nn.Linear(num_sensors + dim, 10),
            nn.Linear(10,num_sensors)
        )

    def forward(self, img, hidden):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 s d -> b s d', b = b)
        hidden = torch.reshape(hidden,(1,self.num_sensors))
        #hidden = torch.reshape(hidden,(1,1,self.dim))
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # print(hidden.shape)
        # print(x.shape)
        input_mlp = torch.cat((hidden,x), dim=1)
        hidden = x
        #return torch.squeeze(self.mlp_head(x),dim=2)
        # print(input_mlp.shape)
        return self.mlp_head(torch.flatten(input_mlp,start_dim=1))