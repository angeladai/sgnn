#include <torch/extension.h>

#include <vector>
#include <cmath>
#include <fstream>

#include "tables.h"
#include "sparsegrid3.h"

#define VOXELSIZE 1.0f

struct vec3f {
	vec3f() {
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}
	vec3f(float x_, float y_, float z_) {
		x = x_;
		y = y_;
		z = z_;
	}
	inline vec3f operator+(const vec3f& other) const {
		return vec3f(x+other.x, y+other.y, z+other.z);
	}
	inline vec3f operator-(const vec3f& other) const {
		return vec3f(x-other.x, y-other.y, z-other.z);
	}
	inline vec3f operator*(float val) const {
		return vec3f(x*val, y*val, z*val);
	}
	inline void operator+=(const vec3f& other) {
		x += other.x;
		y += other.y;
		z += other.z;
	}
	static float distSq(const vec3f& v0, const vec3f& v1) {
		return ((v0.x-v1.x)*(v0.x-v1.x) + (v0.y-v1.y)*(v0.y-v1.y) + (v0.z-v1.z)*(v0.z-v1.z));
	}
	float x;
	float y;
	float z;
};
inline vec3f operator*(float s, const vec3f& v) {
	return v * s;
}
struct vec3uc {
	vec3uc() {
		x = 0;
		y = 0;
		z = 0;
	}
	vec3uc(unsigned char x_, unsigned char y_, unsigned char z_) {
		x = x_;
		y = y_;
		z = z_;
	}
	unsigned char x;
	unsigned char y;
	unsigned char z;
};

struct Triangle {
	vec3f v0;
	vec3f v1;
	vec3f v2;
	vec3uc c0;
	vec3uc c1;
	vec3uc c2;
};

void get_voxel(
	const vec3f& pos,
	at::TensorAccessor<float, 3ul, at::DefaultPtrTraits, long int>& tsdf_accessor,
	at::TensorAccessor<unsigned char, 4ul, at::DefaultPtrTraits, long int>& color_accessor,
	float truncation, 
	float& d, 
	int& w,
	vec3uc& c) {
	int x = (int)round(pos.x);
	int y = (int)round(pos.y);
	int z = (int)round(pos.z);
	if (z >= 0 && z < tsdf_accessor.size(0) && 
		y >= 0 && y < tsdf_accessor.size(1) && 
		x >= 0 && x < tsdf_accessor.size(2)) {
		d = tsdf_accessor[z][y][x];
		c = vec3uc(color_accessor[z][y][x][0], color_accessor[z][y][x][1], color_accessor[z][y][x][2]);
		if (d != -std::numeric_limits<float>::infinity() && fabs(d) < truncation) w = 1;
		else w = 0;
	}
	else {
		d = -std::numeric_limits<float>::infinity();
		w = 0;
		c = vec3uc(0, 0, 0);
	}
	
	// //debugging
	// if ((x == 33 && y == 56 && z == 2) || (x == 2 && y == 56 && z == 33)) {
	// 	bool inbounds = z >= 0 && z < tsdf_accessor.size(0) && 
	// 	y >= 0 && y < tsdf_accessor.size(1) && 
	// 	x >= 0 && x < tsdf_accessor.size(2);
	// 	printf("get_voxel(%f, %f, %f) -> (%d, %d, %d) -> d %f, w %d, c %d %d %d | inbounds %d\n", pos.x, pos.y, pos.z, x, y, z, d, w, (int)c.x, (int)c.y, (int)c.z, (int)inbounds);
	// }
	// //debugging
}

bool trilerp(
	const vec3f& pos, 
	float& dist, 
	vec3uc& color,
	at::TensorAccessor<float, 3ul, at::DefaultPtrTraits, long int>& tsdf_accessor, 
	at::TensorAccessor<unsigned char, 4ul, at::DefaultPtrTraits, long int>& color_accessor,
	float truncation)  {
	const float oSet = VOXELSIZE;
	const vec3f posDual = pos - vec3f(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f);
	vec3f weight = vec3f(pos.x - (int)pos.x, pos.y - (int)pos.y, pos.z - (int)pos.z);

	dist = 0.0f;
	vec3f colorFloat = vec3f(0.0f, 0.0f, 0.0f);
	float d; int w; vec3uc c; vec3f vColor;
	get_voxel(posDual + vec3f(0.0f, 0.0f, 0.0f), tsdf_accessor, color_accessor, truncation, d, w, c); if (w == 0) return false; vColor = vec3f(c.x, c.y, c.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*d; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*vColor;
	get_voxel(posDual + vec3f(oSet, 0.0f, 0.0f), tsdf_accessor, color_accessor, truncation, d, w, c); if (w == 0) return false; vColor = vec3f(c.x, c.y, c.z); dist += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*d; colorFloat += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*vColor;
	get_voxel(posDual + vec3f(0.0f, oSet, 0.0f), tsdf_accessor, color_accessor, truncation, d, w, c); if (w == 0) return false; vColor = vec3f(c.x, c.y, c.z); dist += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*d; colorFloat += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*vColor;
	get_voxel(posDual + vec3f(0.0f, 0.0f, oSet), tsdf_accessor, color_accessor, truncation, d, w, c); if (w == 0) return false; vColor = vec3f(c.x, c.y, c.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *d; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *vColor;
	get_voxel(posDual + vec3f(oSet, oSet, 0.0f), tsdf_accessor, color_accessor, truncation, d, w, c); if (w == 0) return false; vColor = vec3f(c.x, c.y, c.z); dist += weight.x *	   weight.y *(1.0f - weight.z)*d; colorFloat += weight.x *	   weight.y *(1.0f - weight.z)*vColor;
	get_voxel(posDual + vec3f(0.0f, oSet, oSet), tsdf_accessor, color_accessor, truncation, d, w, c); if (w == 0) return false; vColor = vec3f(c.x, c.y, c.z); dist += (1.0f - weight.x)*	   weight.y *	   weight.z *d; colorFloat += (1.0f - weight.x)*	   weight.y *	   weight.z *vColor;
	get_voxel(posDual + vec3f(oSet, 0.0f, oSet), tsdf_accessor, color_accessor, truncation, d, w, c); if (w == 0) return false; vColor = vec3f(c.x, c.y, c.z); dist += weight.x *(1.0f - weight.y)*	   weight.z *d; colorFloat += weight.x *(1.0f - weight.y)*	   weight.z *vColor;
	get_voxel(posDual + vec3f(oSet, oSet, oSet), tsdf_accessor, color_accessor, truncation, d, w, c); if (w == 0) return false; vColor = vec3f(c.x, c.y, c.z); dist += weight.x *	   weight.y *	   weight.z *d; colorFloat += weight.x *	   weight.y *	   weight.z *vColor;
	color = vec3uc(std::round(colorFloat.x), std::round(colorFloat.y), std::round(colorFloat.z));
	return true;
}

vec3f vertexInterp(float isolevel, const vec3f& p1, const vec3f& p2, float d1, float d2)
{
	vec3f r1 = p1;
	vec3f r2 = p2;
    //printf("[interp] r1 = (%f, %f, %f), r2 = (%f, %f, %f) d1 = %f, d2 = %f, iso = %f\n", r1.x, r1.y, r1.z, r2.x, r2.y, r2.z, d1, d2, isolevel);
	//printf("%d, %d, %d || %f, %f, %f -> %f, %f, %f\n", fabs(isolevel - d1) < 0.00001f, fabs(isolevel - d2) < 0.00001f, fabs(d1 - d2) < 0.00001f, isolevel - d1, isolevel - d2, d1-d2, fabs(isolevel - d1), fabs(isolevel - d2), fabs(d1-d2));

	if (fabs(isolevel - d1) < 0.00001f)		return r1;
	if (fabs(isolevel - d2) < 0.00001f)		return r2;
	if (fabs(d1 - d2) < 0.00001f)			return r1;

	float mu = (isolevel - d1) / (d2 - d1);

	vec3f res;
	res.x = p1.x + mu * (p2.x - p1.x); // Positions
	res.y = p1.y + mu * (p2.y - p1.y);
	res.z = p1.z + mu * (p2.z - p1.z);
	
	//printf("[interp] mu = %f, res = (%f, %f, %f)     r1 = (%f, %f, %f), r2 = (%f, %f, %f)\n", mu, res.x, res.y, res.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z);

	return res;
}

void extract_isosurface_at_position(
    const vec3f& pos, 
	at::TensorAccessor<float, 3ul, at::DefaultPtrTraits, long int>& tsdf_accessor,
	at::TensorAccessor<unsigned char, 4ul, at::DefaultPtrTraits, long int>& color_accessor,
	float truncation,
	float isolevel,
	float thresh,
	std::vector<Triangle>& results) {
	const float voxelsize = VOXELSIZE;
	const float P = voxelsize / 2.0f;
	const float M = -P;

    //const bool debugprint = (pos.z == 33 && pos.y == 56 && pos.x == 2) || (pos.z == 2 && pos.y == 56 && pos.x == 33);

	vec3f p000 = pos + vec3f(M, M, M); float dist000; vec3uc color000; bool valid000 = trilerp(p000, dist000, color000, tsdf_accessor, color_accessor, truncation);
	vec3f p100 = pos + vec3f(P, M, M); float dist100; vec3uc color100; bool valid100 = trilerp(p100, dist100, color100, tsdf_accessor, color_accessor, truncation);
	vec3f p010 = pos + vec3f(M, P, M); float dist010; vec3uc color010; bool valid010 = trilerp(p010, dist010, color010, tsdf_accessor, color_accessor, truncation);
	vec3f p001 = pos + vec3f(M, M, P); float dist001; vec3uc color001; bool valid001 = trilerp(p001, dist001, color001, tsdf_accessor, color_accessor, truncation);
	vec3f p110 = pos + vec3f(P, P, M); float dist110; vec3uc color110; bool valid110 = trilerp(p110, dist110, color110, tsdf_accessor, color_accessor, truncation);
	vec3f p011 = pos + vec3f(M, P, P); float dist011; vec3uc color011; bool valid011 = trilerp(p011, dist011, color011, tsdf_accessor, color_accessor, truncation);
	vec3f p101 = pos + vec3f(P, M, P); float dist101; vec3uc color101; bool valid101 = trilerp(p101, dist101, color101, tsdf_accessor, color_accessor, truncation);
	vec3f p111 = pos + vec3f(P, P, P); float dist111; vec3uc color111; bool valid111 = trilerp(p111, dist111, color111, tsdf_accessor, color_accessor, truncation);
	//if (debugprint) {
	//	printf("[extract_isosurface_at_position] pos: %f, %f, %f\n", pos.x, pos.y, pos.z);
	//	printf("\tp000 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p000.x, p000.y, p000.z, dist000, (int)color000.x, (int)color000.y, (int)color000.z, (int)valid000);
	//	printf("\tp100 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p100.x, p100.y, p100.z, dist100, (int)color100.x, (int)color100.y, (int)color100.z, (int)valid100);
	//	printf("\tp010 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p010.x, p010.y, p010.z, dist010, (int)color010.x, (int)color010.y, (int)color010.z, (int)valid010);
	//	printf("\tp001 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p001.x, p001.y, p001.z, dist001, (int)color001.x, (int)color001.y, (int)color001.z, (int)valid001);
	//	printf("\tp110 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p110.x, p110.y, p110.z, dist110, (int)color110.x, (int)color110.y, (int)color110.z, (int)valid110);
	//	printf("\tp011 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p011.x, p011.y, p011.z, dist011, (int)color011.x, (int)color011.y, (int)color011.z, (int)valid011);
	//	printf("\tp101 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p101.x, p101.y, p101.z, dist101, (int)color101.x, (int)color101.y, (int)color101.z, (int)valid101);
	//	printf("\tp111 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p111.x, p111.y, p111.z, dist111, (int)color111.x, (int)color111.y, (int)color111.z, (int)valid111);
	//}
	if (!valid000 || !valid100 || !valid010 || !valid001 || !valid110 || !valid011 || !valid101 || !valid111) return;
	
	uint cubeindex = 0;
	if (dist010 < isolevel) cubeindex += 1;
	if (dist110 < isolevel) cubeindex += 2;
	if (dist100 < isolevel) cubeindex += 4;
	if (dist000 < isolevel) cubeindex += 8;
	if (dist011 < isolevel) cubeindex += 16;
	if (dist111 < isolevel) cubeindex += 32;
	if (dist101 < isolevel) cubeindex += 64;
	if (dist001 < isolevel) cubeindex += 128;
	const float thres = thresh;
	float distArray[] = { dist000, dist100, dist010, dist001, dist110, dist011, dist101, dist111 };
	//if (debugprint) {
	//	printf("dists (%f, %f, %f, %f, %f, %f, %f, %f)\n", dist000, dist100, dist010, dist001, dist110, dist011, dist101, dist111);
	//	printf("cubeindex %d\n", cubeindex);
	//}
	for (uint k = 0; k < 8; k++) {
		for (uint l = 0; l < 8; l++) {
			if (distArray[k] * distArray[l] < 0.0f) {
				if (fabs(distArray[k]) + fabs(distArray[l]) > thres) return;
			}
			else {
				if (fabs(distArray[k] - distArray[l]) > thres) return;
			}
		}
	}
	if (fabs(dist000) > thresh) return;
	if (fabs(dist100) > thresh) return;
	if (fabs(dist010) > thresh) return;
	if (fabs(dist001) > thresh) return;
	if (fabs(dist110) > thresh) return;
	if (fabs(dist011) > thresh) return;
	if (fabs(dist101) > thresh) return;
	if (fabs(dist111) > thresh) return;
	
	if (edgeTable[cubeindex] == 0 || edgeTable[cubeindex] == 255) return; // added by me edgeTable[cubeindex] == 255

	vec3uc c;
	{
		float d; int w; 
		get_voxel(pos, tsdf_accessor, color_accessor, truncation, d, w, c); 
	}
	
	vec3f vertlist[12];
	if (edgeTable[cubeindex] & 1)	    vertlist[0] = vertexInterp(isolevel, p010, p110, dist010, dist110);
	if (edgeTable[cubeindex] & 2)	    vertlist[1] = vertexInterp(isolevel, p110, p100, dist110, dist100);
	if (edgeTable[cubeindex] & 4)	    vertlist[2] = vertexInterp(isolevel, p100, p000, dist100, dist000);
	if (edgeTable[cubeindex] & 8)	    vertlist[3] = vertexInterp(isolevel, p000, p010, dist000, dist010);
	if (edgeTable[cubeindex] & 16)	vertlist[4] = vertexInterp(isolevel, p011, p111, dist011, dist111);
	if (edgeTable[cubeindex] & 32)	vertlist[5] = vertexInterp(isolevel, p111, p101, dist111, dist101);
	if (edgeTable[cubeindex] & 64)	vertlist[6] = vertexInterp(isolevel, p101, p001, dist101, dist001);
	if (edgeTable[cubeindex] & 128)	vertlist[7] = vertexInterp(isolevel, p001, p011, dist001, dist011);
	if (edgeTable[cubeindex] & 256)	vertlist[8] = vertexInterp(isolevel, p010, p011, dist010, dist011);
	if (edgeTable[cubeindex] & 512)	vertlist[9] = vertexInterp(isolevel, p110, p111, dist110, dist111);
	if (edgeTable[cubeindex] & 1024)  vertlist[10] = vertexInterp(isolevel, p100, p101, dist100, dist101);
	if (edgeTable[cubeindex] & 2048)  vertlist[11] = vertexInterp(isolevel, p000, p001, dist000, dist001);

	for (int i = 0; triTable[cubeindex][i] != -1; i += 3)
	{
		Triangle t;
		t.v0 = vertlist[triTable[cubeindex][i + 0]];
		t.v1 = vertlist[triTable[cubeindex][i + 1]];
		t.v2 = vertlist[triTable[cubeindex][i + 2]];
		t.c0 = c;
		t.c1 = c;
		t.c2 = c;

        //printf("triangle at (%f, %f, %f): (%f, %f, %f) (%f, %f, %f) (%f, %f, %f)\n", pos.x, pos.y, pos.z, t.v0.x, t.v0.y, t.v0.z, t.v1.x, t.v1.y, t.v1.z, t.v2.x, t.v2.y, t.v2.z);
		//printf("vertlist idxs: %d, %d, %d (%d, %d, %d)\n", triTable[cubeindex][i + 0], triTable[cubeindex][i + 1], triTable[cubeindex][i + 2], edgeTable[cubeindex] & 1, edgeTable[cubeindex] & 256, edgeTable[cubeindex] & 8);
		//getchar();
		results.push_back(t);
	}
}


// ----- MESH CLEANUP FUNCTIONS
unsigned int remove_duplicate_faces(std::vector<vec3i>& faces)
{
	struct vecHash {
		size_t operator()(const std::vector<unsigned int>& v) const {
			//TODO larger prime number (64 bit) to match size_t
			const size_t p[] = {73856093, 19349669, 83492791};
			size_t res = 0;
			for (unsigned int i : v) {
				res = res ^ (size_t)i * p[i%3];
			}
			return res;
			//const size_t res = ((size_t)v.x * p0)^((size_t)v.y * p1)^((size_t)v.z * p2);
		}
	};

	size_t numFaces = faces.size();
	std::vector<vec3i> new_faces;	new_faces.reserve(numFaces);

	std::unordered_set<std::vector<unsigned int>, vecHash> _set;
	for (size_t i = 0; i < numFaces; i++) {
		std::vector<unsigned int> face = {(unsigned int)faces[i].x, (unsigned int)faces[i].y, (unsigned int)faces[i].z};
		std::sort(face.begin(), face.end());
		if (_set.find(face) == _set.end()) {
			//not found yet
			_set.insert(face);
			new_faces.push_back(faces[i]);	//inserted the unsorted one
		}
	}
	if (faces.size() != new_faces.size()) {
		faces = new_faces;
	}
	//printf("Removed %d-%d=%d duplicate faces of %d\n", (int)numFaces, (int)new_faces.size(), (int)numFaces-(int)new_faces.size(), (int)numFaces);

	return (unsigned int)new_faces.size();
}
unsigned int remove_degenerate_faces(std::vector<vec3i>& faces)
{
	std::vector<vec3i> new_faces;

	for (size_t i = 0; i < faces.size(); i++) {
		std::unordered_set<int> _set(3);
		bool foundDuplicate = false;
		if (_set.find(faces[i].x) != _set.end()) { foundDuplicate = true; } 
		else { _set.insert(faces[i].x); }                                 
		if (!foundDuplicate && _set.find(faces[i].y) != _set.end()) { foundDuplicate = true; } 
		else { _set.insert(faces[i].y); }                                 
		if (!foundDuplicate && _set.find(faces[i].z) != _set.end()) { foundDuplicate = true; } 
		else { _set.insert(faces[i].z); }
		if (!foundDuplicate) {
			new_faces.push_back(faces[i]);
		}
	}
	if (faces.size() != new_faces.size()) {
		faces = new_faces;
	}

	return (unsigned int)faces.size();
}
unsigned int hasNearestNeighbor( const vec3i& coord, SparseGrid3<std::list<std::pair<vec3f,unsigned int> > > &neighborQuery, const vec3f& v, float thresh )
{
	float threshSq = thresh*thresh;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				vec3i c = coord + vec3i(i,j,k);
				if (neighborQuery.exists(c)) {
					for (const std::pair<vec3f, unsigned int>& n : neighborQuery[c]) {
						if (vec3f::distSq(v,n.first) < threshSq) {
							return n.second;
						}
					}
				}
			}
		}
	}
	return (unsigned int)-1;
}
unsigned int hasNearestNeighborApprox(const vec3i& coord, SparseGrid3<unsigned int> &neighborQuery) {
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				vec3i c = coord + vec3i(i,j,k);
				if (neighborQuery.exists(c)) {
					return neighborQuery[c];
				}
			}
		}
	}
	return (unsigned int)-1;
}
int sgn(float val) {
    return (0.0f < val) - (val < 0.0f);
}
std::pair<std::pair<std::vector<vec3f>,std::vector<vec3uc>>, std::vector<vec3i>> merge_close_vertices(const std::vector<Triangle>& meshTris, float thresh, bool approx)
{
	// assumes voxelsize = 1
	assert(thresh > 0);
	unsigned int numV = (unsigned int)meshTris.size() * 3;
	std::vector<vec3f> vertices(numV);
	std::vector<vec3uc> colors(numV);
	std::vector<vec3i> faces(meshTris.size());
	for (int i = 0; i < (int)meshTris.size(); i++) {
		vertices[3*i+0].x = meshTris[i].v0.x;
		vertices[3*i+0].y = meshTris[i].v0.y;
		vertices[3*i+0].z = meshTris[i].v0.z;
		
		vertices[3*i+1].x = meshTris[i].v1.x;
		vertices[3*i+1].y = meshTris[i].v1.y;
		vertices[3*i+1].z = meshTris[i].v1.z;
		
		vertices[3*i+2].x = meshTris[i].v2.x;
		vertices[3*i+2].y = meshTris[i].v2.y;
		vertices[3*i+2].z = meshTris[i].v2.z;

		colors[3*i+0].x = meshTris[i].c0.x;
		colors[3*i+0].y = meshTris[i].c0.y;
		colors[3*i+0].z = meshTris[i].c0.z;
		
		colors[3*i+1].x = meshTris[i].c1.x;
		colors[3*i+1].y = meshTris[i].c1.y;
		colors[3*i+1].z = meshTris[i].c1.z;
		
		colors[3*i+2].x = meshTris[i].c2.x;
		colors[3*i+2].y = meshTris[i].c2.y;
		colors[3*i+2].z = meshTris[i].c2.z;
		
		faces[i].x = 3*i+0;
		faces[i].y = 3*i+1;
		faces[i].z = 3*i+2;
	}

	std::vector<unsigned int> vertexLookUp;	vertexLookUp.resize(numV);
	std::vector<vec3f> new_verts; new_verts.reserve(numV);
	std::vector<vec3uc> new_colors; new_colors.reserve(numV);

	unsigned int cnt = 0;
	if (approx) {
		SparseGrid3<unsigned int> neighborQuery(0.6f, numV*2);
		for (unsigned int v = 0; v < numV; v++) {

			const vec3f& vert = vertices[v];
			vec3i coord = vec3i(vert.x/thresh + 0.5f*sgn(vert.x), vert.y/thresh + 0.5f*sgn(vert.y), vert.z/thresh + 0.5f*sgn(vert.z));			
			unsigned int nn = hasNearestNeighborApprox(coord, neighborQuery);

			if (nn == (unsigned int)-1) {
				neighborQuery[coord] = cnt;
				new_verts.push_back(vert);
				new_colors.push_back(colors[v]);
				vertexLookUp[v] = cnt;
				cnt++;
			} else {
				vertexLookUp[v] = nn;
			}
		}
	} else {
		SparseGrid3<std::list<std::pair<vec3f, unsigned int> > > neighborQuery(0.6f, numV*2);
		for (unsigned int v = 0; v < numV; v++) {

			const vec3f& vert = vertices[v];
			vec3i coord = vec3i(vert.x/thresh + 0.5f*sgn(vert.x), vert.y/thresh + 0.5f*sgn(vert.y), vert.z/thresh + 0.5f*sgn(vert.z));
			unsigned int nn = hasNearestNeighbor(coord, neighborQuery, vert, thresh);

			if (nn == (unsigned int)-1) {
				neighborQuery[coord].push_back(std::make_pair(vert,cnt));
				new_verts.push_back(vert);
				new_colors.push_back(colors[v]);
				vertexLookUp[v] = cnt;
				cnt++;
			} else {
				vertexLookUp[v] = nn;
			}
		}
	}
	// Update faces
	for (int i = 0; i < (int)faces.size(); i++) {		
		faces[i].x = vertexLookUp[faces[i].x];
		faces[i].y = vertexLookUp[faces[i].y];
		faces[i].z = vertexLookUp[faces[i].z];
	}

	if (vertices.size() != new_verts.size()) {
		vertices = new_verts;
	}
	if (colors.size() != new_colors.size()) {
		colors = new_colors;
	}

	remove_degenerate_faces(faces);
	//printf("Merged %d-%d=%d of %d vertices\n", numV, cnt, numV-cnt, numV);
	return std::make_pair(std::make_pair(vertices, colors), faces);
}
// ----- MESH CLEANUP FUNCTIONS

void run_marching_cubes_internal(
    at::Tensor tsdf,
    at::Tensor colors,
	float isovalue,
	float truncation,
	float thresh,
	std::vector<Triangle>& results) {
	results.clear();
	
	auto tsdf_accessor = tsdf.accessor<float,3>();
	auto color_accessor = colors.accessor<unsigned char,4>();
	for (int i = 0; i < (int)tsdf.size(0); i++) {
		for (int j = 0; j < (int)tsdf.size(1); j++) {
			for (int k = 0; k < (int)tsdf.size(2); k++) {
			extract_isosurface_at_position(vec3f(k, j, i), tsdf_accessor, color_accessor, truncation, isovalue, thresh, results);
			} // k
		} // j
	} // i
	//printf("#results = %d\n", (int)results.size());
}

std::vector<at::Tensor> run_marching_cubes(
    at::Tensor tsdf,
    at::Tensor colors,
	float isovalue,
	float truncation,
	float thresh) {
	std::vector<Triangle> results;	
	run_marching_cubes_internal(tsdf, colors, isovalue, truncation, thresh, results);
	
	// cleanup
	auto cleaned = merge_close_vertices(results, 0.00001f, true);
	remove_duplicate_faces(cleaned.second);
	
	// triangle soup
	const unsigned int numV = cleaned.first.first.size();
	const unsigned int numF = cleaned.second.size();
	at::Tensor vertices = torch::zeros({numV, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(tsdf.device()));
	at::Tensor vertex_colors = torch::zeros({numV, 3}, torch::TensorOptions().dtype(torch::kByte).device(tsdf.device()));
	at::Tensor faces = torch::zeros({numF, 3}, torch::TensorOptions().dtype(torch::kInt32).device(tsdf.device()));
	auto vertices_accessor = vertices.accessor<float,2>();
	auto vertex_colors_accessor = vertex_colors.accessor<unsigned char,2>();
	auto faces_accessor = faces.accessor<int,2>();
	for (int i = 0; i < (int)numV; i++) {
		vertices_accessor[i][0] = cleaned.first.first[i].x;
		vertices_accessor[i][1] = cleaned.first.first[i].y;
		vertices_accessor[i][2] = cleaned.first.first[i].z;
		vertex_colors_accessor[i][0] = cleaned.first.second[i].x;
		vertex_colors_accessor[i][1] = cleaned.first.second[i].y;
		vertex_colors_accessor[i][2] = cleaned.first.second[i].z;
	}
	for (int i = 0; i < (int)cleaned.second.size(); i++) {		
		faces_accessor[i][0] = cleaned.second[i].x;
		faces_accessor[i][1] = cleaned.second[i].y;
		faces_accessor[i][2] = cleaned.second[i].z;
	}
	//printf("#vertices = (%d, %d), #faces = (%d, %d)\n", vertices.size(0), vertices.size(1), faces.size(0), faces.size(1));
	return {vertices, vertex_colors, faces};
}

void save_to_ply(const std::string& filename,
	const std::vector<vec3f>& verts,
	const std::vector<vec3uc>& vertcolors,
	const std::vector<vec3i>& indices) {
	const unsigned int numV = (unsigned int)verts.size();
	const unsigned int numF = (unsigned int)indices.size();
	//std::cout << "[save_to_ply] " << filename << " (" << std::to_string(numV) << ", " << std::to_string(numF) << ")" << std::endl;
	std::ofstream file(filename, std::ios::binary);
	file << "ply\n";
	file << "format binary_little_endian 1.0\n";
	file << "element vertex " << numV << "\n";
	file << "property float x\n";
	file << "property float y\n";
	file << "property float z\n";
	file << "property uchar red\n";
	file << "property uchar green\n";
	file << "property uchar blue\n";
	//file << "property uchar alpha\n";
	file << "element face " << numF << "\n";
	file << "property list uchar int vertex_indices\n";
	file << "end_header\n";

	{
		size_t vertexByteSize = sizeof(float)*3 + sizeof(unsigned char)*3;
		unsigned char* data = new unsigned char[vertexByteSize*numV];
		size_t byteOffset = 0;
		for (size_t i = 0; i < numV; i++) {
			memcpy(&data[byteOffset], &verts[i], sizeof(float)*3);
			byteOffset += sizeof(float)*3;
			memcpy(&data[byteOffset], &vertcolors[i], sizeof(unsigned char)*3);
			byteOffset += sizeof(unsigned char)*3;
		}
		file.write((const char*)data, byteOffset);
		if (data) { delete[] data;   data=nullptr; }
	}
	const unsigned char numFaceIndices = 3;
	for (size_t i = 0; i < numF; i++) {
		file.write((const char*)&numFaceIndices, sizeof(unsigned char));
		file.write((const char*)&indices[i], numFaceIndices*sizeof(int));
	}
	file.close();
}

void export_marching_cubes(
    at::Tensor tsdf,
    at::Tensor colors,
	float isovalue,
	float truncation,
	float thresh,
	const std::string& filename) {
	std::vector<Triangle> results;	
	run_marching_cubes_internal(tsdf, colors, isovalue, truncation, thresh, results);
	
	// cleanup
	auto cleaned = merge_close_vertices(results, 0.00001f, true);
	remove_duplicate_faces(cleaned.second);
	
	save_to_ply(filename, cleaned.first.first, cleaned.first.second, cleaned.second);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_marching_cubes", &run_marching_cubes, "Marching Cubes");
  m.def("export_marching_cubes", &export_marching_cubes, "Marching Cubes + Save to PLY File");
}
